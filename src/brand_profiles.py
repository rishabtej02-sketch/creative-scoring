#!/usr/bin/env python3
"""
brand_profiles.py v2 - reference pools for percentile ranking.

CHANGES FROM v1 (driven by the 00_inspect.py output):
  1. youtube_scores.csv has NO brand column. Brand is now joined in from
     features.csv (video_id -> channel_handle), with the display name
     resolved through channels_resolved.csv (handle -> channel_title).
  2. Duplicate creatives are removed. Google Ads Transparency re-lists the
     same creative under many ad ids, which produced pools where
     p10 == p50 (400+ Zomato ads sharing one score to 6 decimals).
     The scorer is deterministic, so an exact score collision inside one
     brand means identical pixels. That is the dedup key.
  3. Degenerate pools are now detected and reported, not silently written,
     and the percentile they return is flagged unreliable.

BUILD:
    python src/brand_profiles.py --audit                # dup report first
    python src/brand_profiles.py --build
    python src/brand_profiles.py --build --no-dedup     # before/after compare
    python src/brand_profiles.py --show zomato

USE:
    from brand_profiles import BrandProfiles, allocate_budget
    bp = BrandProfiles.load()
    r = bp.percentile(0.52, platform="meta", brand="zomato")
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "brand_profiles.json")

GRID = np.arange(0, 101)
MIN_UNIQUE_FRAC = 0.5   # degenerate if distinct scores < 50% of rows


def norm_brand(x) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\.(com|in|co)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or "unknown"


# ---------------------------------------------------------------- loaders
def _load_channel_display() -> dict:
    """handle -> pretty title, e.g. 'boat-lifestyle' -> 'boAt'."""
    p = os.path.join(DATA, "channels_resolved.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        title = str(r.get("channel_title", "")).strip()
        if not title:
            continue
        for k in ("handle", "channel_id", "channel_title"):
            if k in df.columns and pd.notna(r.get(k)):
                out[norm_brand(r[k])] = title
    return out


def load_youtube(verbose=True) -> pd.DataFrame:
    sp = os.path.join(DATA, "youtube_scores.csv")
    fp = os.path.join(DATA, "features.csv")
    if not os.path.exists(sp):
        print("  skip: youtube_scores.csv missing")
        return pd.DataFrame()
    if not os.path.exists(fp):
        print("  !! features.csv missing - cannot resolve YouTube brands")
        return pd.DataFrame()

    sc = pd.read_csv(sp)
    feat = pd.read_csv(fp, usecols=lambda c: c in
                       ("video_id", "channel_handle", "channel_id"))
    m = sc.merge(feat, on="video_id", how="left")
    miss = int(m["channel_handle"].isna().sum())

    disp = _load_channel_display()
    out = pd.DataFrame({
        "id": m["video_id"].astype(str),
        "brand_raw": m["channel_handle"].astype(str),
        "brand": m["channel_handle"].map(norm_brand),
        "score": pd.to_numeric(m["score"], errors="coerce"),
        "label": pd.to_numeric(m.get("overperformed"), errors="coerce"),
        "platform": "youtube",
    })
    out["display"] = out["brand"].map(lambda b: disp.get(b, ""))
    out["labelable"] = (m["labelable"].astype(str).str.lower().isin(["true", "1"])
                        if "labelable" in m.columns else True)
    out = out[out["brand"] != "unknown"].dropna(subset=["score"])

    if verbose:
        lab = out.loc[out["labelable"], "label"].dropna()
        print(f"  youtube_scores + features.csv      rows={len(out):<7} "
              f"brands={out['brand'].nunique():<4} unjoined={miss}")
        print(f"     labelable={int(out['labelable'].sum()):<6} "
              f"label_rate={lab.mean():.3f}" if len(lab) else "")
    return out


def load_paid(verbose=True) -> pd.DataFrame:
    p = os.path.join(DATA, "paid_ad_scores.csv")
    if not os.path.exists(p):
        print("  skip: paid_ad_scores.csv missing")
        return pd.DataFrame()
    df = pd.read_csv(p)
    out = pd.DataFrame({
        "id": df["creative_id"].astype(str),
        "brand_raw": df["brand"].astype(str),
        "brand": df["brand"].map(norm_brand),
        "score": pd.to_numeric(df["score"], errors="coerce"),
        "label": np.nan,
        "platform": df["source"].astype(str).str.lower().str.strip(),
        "display": "",
        "labelable": False,
    }).dropna(subset=["score"])
    if verbose:
        print(f"  paid_ad_scores.csv                 rows={len(out):<7} "
              f"{out['platform'].value_counts().to_dict()}")
    return out


def load_all(verbose=True) -> pd.DataFrame:
    parts = [d for d in (load_youtube(verbose), load_paid(verbose)) if len(d)]
    if not parts:
        sys.exit("No usable score files. Run 00_inspect.py.")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- dedup
DEDUP_PLATFORMS = {"google", "meta"}
HASH_FILE = os.path.join(DATA, "clip_paid_meta.csv")


def _hash_map():
    """id -> md5 of the image file, written by embed_paid.py. This is TRUE
    duplicate detection. Score-tie dedup below is only a fallback."""
    if not os.path.exists(HASH_FILE):
        return None
    df = pd.read_csv(HASH_FILE)
    if "md5" not in df.columns:
        return None
    return dict(zip(df["id"].astype(str), df["md5"].astype(str)))


def dedup(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Remove repeat listings of the same creative.

    Only applied to Google/Meta. YouTube video_ids are unique by construction,
    and isotonic calibration makes ~18% of YouTube scores tie legitimately, so
    a score-based key there would delete real rows.

    Key preference:
      1. md5 of the image file (exact, from embed_paid.py) - correct.
      2. score rounded to 6dp within brand - fallback. The scorer is
         deterministic, so identical pixels give identical scores, but
         isotonic calibration also produces genuine ties, which means this
         key OVER-deletes. Treat its counts as an upper bound.
    """
    hm = _hash_map()
    before = len(df)
    d = df.copy()
    d["_paid"] = d["platform"].isin(DEDUP_PLATFORMS)

    if hm:
        d["_k"] = d["id"].astype(str).map(hm)
        d["_k"] = d["_k"].fillna("s" + d["score"].round(6).astype(str))
        method = "md5 (exact)"
    else:
        d["_k"] = "s" + d["score"].round(6).astype(str)
        method = "score-tie (fallback, over-deletes)"

    paid = d[d["_paid"]].drop_duplicates(subset=["platform", "brand", "_k"],
                                         keep="first")
    out = pd.concat([d[~d["_paid"]], paid], ignore_index=True)
    out = out.drop(columns=["_k", "_paid"])
    if verbose:
        print(f"\n  dedup method: {method}   (YouTube exempt)")
        rem = before - len(out)
        print(f"  dedup: {before} -> {len(out)}  removed={rem} "
              f"({100*rem/max(before,1):.1f}%)")
        g = (d[d["_paid"]].groupby(["platform", "brand"])
             .agg(n=("_k", "size"), u=("_k", "nunique")))
        g["dropped"] = g["n"] - g["u"]
        top = g.sort_values("dropped", ascending=False)
        top = top[top["dropped"] > 0].head(10)
        if len(top):
            print("  worst duplicate offenders:")
            for (pl, br), r in top.iterrows():
                print(f"    {pl:<8} {br:<24} {int(r.n):>6} -> {int(r.u):<6} "
                      f"(-{int(r.dropped)})")
    return out


# ---------------------------------------------------------------- stats
def _stats(scores) -> dict:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    g = [float(v) for v in np.percentile(s, GRID)]
    nu = int(np.unique(s).size)
    return {
        "n": int(s.size), "n_unique": nu,
        "mean": float(s.mean()), "std": float(s.std(ddof=0)),
        "min": float(s.min()), "max": float(s.max()),
        "grid": g,
        "degenerate": bool(g[10] == g[50] or nu < MIN_UNIQUE_FRAC * s.size),
    }


def build(min_n=30, do_dedup=True):
    print("Building brand profiles...")
    df = load_all()
    if do_dedup:
        df = dedup(df)
    df = df[df["platform"].isin(["youtube", "meta", "google"])]

    disp_map = {}
    for b, g in df.groupby("brand"):
        d = g.loc[g["display"].astype(str).str.len() > 0, "display"]
        disp_map[b] = d.iloc[0] if len(d) else g["brand_raw"].value_counts().idxmax()

    pools, flags = {}, []
    print()
    for plat, gp in df.groupby("platform"):
        pools[plat] = {"_global": _stats(gp["score"].values)}
        kept = dropped = 0
        for brand, gb in gp.groupby("brand"):
            if len(gb) < min_n:
                dropped += 1
                continue
            st = _stats(gb["score"].values)
            st["display"] = disp_map.get(brand, brand)
            lab = gb.loc[gb["labelable"] == True, "label"].dropna()
            st["label_rate"] = float(lab.mean()) if len(lab) else None
            if st["degenerate"]:
                flags.append((plat, brand, st["n"], st["n_unique"]))
            pools[plat][brand] = st
            kept += 1
        g = pools[plat]["_global"]
        print(f"  {plat:<9} n={g['n']:<6} uniq={g['n_unique']:<6} "
              f"mean={g['mean']:.3f} sd={g['std']:.3f} "
              f"| brands kept={kept} dropped(<{min_n})={dropped}")

    if flags:
        print(f"\n  !! {len(flags)} pool(s) still degenerate after dedup:")
        for pl, br, n, nu in flags[:12]:
            print(f"     {pl:<8} {br:<24} n={n:<5} unique={nu}")
        print("     -> near-identical creatives; percentile flagged unreliable.")

    doc = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_n_brand": min_n, "deduped": do_dedup, "grid_points": len(GRID),
        "labeled_platforms": ["youtube"],
        "caveat": ("Only the YouTube pool carries real overperformance labels. "
                   "Meta and Google pools are unlabeled reference distributions; "
                   "percentiles there express similarity to winning creatives, "
                   "not measured performance."),
        "pools": pools,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
    return doc


def audit():
    print("=== AUDIT ===")
    raw = load_all()
    ded = dedup(raw)
    print("\n  per-platform, raw vs deduped:")
    print(f"  {'platform':<10}{'raw':>8}{'dedup':>8}{'dropped':>9}"
          f"{'mean':>9}{'sd_raw':>9}{'sd_ded':>9}")
    for pl in sorted(raw["platform"].unique()):
        a, b = raw[raw.platform == pl], ded[ded.platform == pl]
        if not len(a):
            continue
        print(f"  {pl:<10}{len(a):>8}{len(b):>8}{len(a)-len(b):>9}"
              f"{b.score.mean():>9.3f}{a.score.std():>9.3f}{b.score.std():>9.3f}")
    yt = ded[ded.platform == "youtube"]["score"]
    if len(yt):
        print("\n  compression vs YouTube (sd ratio, post-dedup):")
        for pl in ("meta", "google"):
            s = ded[ded.platform == pl]["score"]
            if len(s):
                print(f"    {pl:<8} sd={s.std():.4f}  yt_sd={yt.std():.4f}  "
                      f"ratio={yt.std()/s.std():.2f}x")


# ---------------------------------------------------------------- lookup API
@dataclass
class Rank:
    pct: float
    basis: str
    n: int
    pool: str
    label: str
    labeled: bool
    caveat: str
    reliable: bool = True

    def sentence(self, brand_display=None) -> str:
        b = brand_display or self.pool
        verb = "beats" if self.labeled else "is more winner-like than"
        return f"{verb} {self.pct:.0f}% of {b} creatives"


def band(pct):
    if pct >= 90: return "Top 10%"
    if pct >= 75: return "Strong"
    if pct >= 55: return "Above average"
    if pct >= 45: return "Average"
    if pct >= 25: return "Below average"
    return "Bottom 25%"


class BrandProfiles:
    def __init__(self, doc):
        self.doc = doc
        self.pools = doc["pools"]
        self.labeled = set(doc.get("labeled_platforms", []))
        self.caveat = doc.get("caveat", "")

    @classmethod
    def load(cls, path=OUT):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run: python src/brand_profiles.py --build")
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def platforms(self):
        return [p for p in ("youtube", "meta", "google") if p in self.pools]

    def brands(self, platform):
        p = self.pools.get(platform, {})
        return sorted([(k, v.get("display", k), v["n"])
                       for k, v in p.items() if not k.startswith("_")],
                      key=lambda t: -t[2])

    def has(self, platform, brand):
        return norm_brand(brand) in self.pools.get(platform, {})

    def percentile(self, score, platform, brand=None) -> Rank:
        pool = self.pools.get(platform)
        if pool is None:
            return Rank(50.0, "none", 0, platform, "Unknown", False,
                        "No reference pool for this platform.", False)
        key = norm_brand(brand) if brand else None
        if key and key in pool:
            e, basis = pool[key], "brand"
            name, caveat = e.get("display", key), ""
        else:
            e, basis = pool["_global"], "platform"
            name = f"all {platform}"
            caveat = (f"No pool for '{brand}' (need >= {self.doc['min_n_brand']} "
                      f"ads). Ranked against all {platform} ads instead.")
        grid = np.asarray(e["grid"], dtype=float)
        pct = float(np.interp(score, grid, GRID.astype(float),
                              left=0.0, right=100.0))
        labeled = platform in self.labeled
        reliable = not e.get("degenerate", False)
        if not labeled:
            caveat += (" " if caveat else "") + (
                f"{platform} pool is unlabeled - similarity to winners, "
                "not measured results.")
        if not reliable:
            caveat += (" " if caveat else "") + (
                "This pool has very few distinct scores (near-duplicate "
                "creatives), so the percentile is unstable.")
        return Rank(round(pct, 1), basis, int(e["n"]), name, band(pct),
                    labeled, caveat.strip(), reliable)

    def context_block(self, platform, brand=None) -> str:
        pool = self.pools.get(platform, {})
        key = norm_brand(brand) if brand else None
        e = pool.get(key) or pool.get("_global")
        if not e:
            return "no reference pool"
        g = e["grid"]
        who = e.get("display", key) if key in pool else f"all {platform}"
        return (f"reference pool = {who} ({platform}, n={e['n']}) | "
                f"p10={g[10]:.3f} p25={g[25]:.3f} p50={g[50]:.3f} "
                f"p75={g[75]:.3f} p90={g[90]:.3f}")


def allocate_budget(pcts, budget, floor_frac=0.05, power=1.0):
    """Split budget on PERCENTILES, not raw scores. Raw paid-ad scores sit in
    a ~0.10-wide band, so proportional splitting on them is near-uniform."""
    p = np.asarray(pcts, dtype=float).clip(0, 100) / 100.0
    n = len(p)
    if n == 0:
        return []
    w = p ** power
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    floor = min(floor_frac, 1.0 / n)
    w = floor + (1 - floor * n) * w
    a = np.round(w * budget, 2)
    a[-1] = round(budget - a[:-1].sum(), 2)
    return a.tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--show", metavar="BRAND")
    a = ap.parse_args()

    if a.audit:
        audit()
    if a.build:
        build(min_n=a.min_n, do_dedup=not a.no_dedup)
    if a.show:
        bp = BrandProfiles.load()
        for pl in bp.platforms():
            if not bp.has(pl, a.show):
                print(f"[{pl}] no pool for {a.show}")
                continue
            e = bp.pools[pl][norm_brand(a.show)]
            g = e["grid"]
            warn = "  << DEGENERATE" if e.get("degenerate") else ""
            print(f"[{pl}] {e.get('display')}  n={e['n']} uniq={e['n_unique']} "
                  f"mean={e['mean']:.3f} sd={e['std']:.3f}{warn}")
            print(f"        p10={g[10]:.3f} p50={g[50]:.3f} p90={g[90]:.3f}")
            for s in (0.40, 0.45, 0.50, 0.55, 0.60):
                r = bp.percentile(s, pl, a.show)
                print(f"        score {s:.2f} -> p{r.pct:5.1f}  {r.label}")
    if not (a.build or a.show or a.audit):
        ap.print_help()
