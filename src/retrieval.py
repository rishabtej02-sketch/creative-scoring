#!/usr/bin/env python3
"""
retrieval.py v2 - CLIP nearest-neighbour search over your own ad corpus.

Rewritten against the confirmed schema (v1 was written blind):
  data/clip_embeddings.npy       (13062, 512)  ids = video_id
  data/clip_embeddings_paid.npy  (N, 512)      ids = creative_id
  data/clip_paid_meta.csv        id, platform, path, md5
  data/youtube_scores.csv        video_id, score, overperformed, labelable
  data/features.csv              video_id, channel_handle, + 9 CV cols
  data/videos_labeled.csv        video_id, title, thumbnail_url
  data/meta_ads_ocr.csv          brand, creative_id, ocr_text, char_len
  data/paid_ad_scores.csv        source, brand, creative_id, score

CRITICAL: the index is md5-deduped. Without it "5 most similar ads" would
return 5 copies of the same image - 61.6% of the Google folder is duplicates,
and one creative appears 866 times. Duplicates would dominate every result.

TWO INDEXES:
  winners : YouTube, labelable==True AND overperformed==1. Real labels.
            -> drives the LLM advice.
  all     : every unique creative, all platforms. No labels.
            -> drives the "similar ads" strip.

BUILD:
    python src/retrieval.py --build
    python src/retrieval.py --stats

USE:
    from retrieval import Retriever
    r = Retriever.load()
    hits = r.search(vec512, k=5, winners_only=True, brand="zomato")
    print(r.prompt_block(hits, own_feats={"text_area_frac": 0.31}))
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
IDX_WINNERS = os.path.join(MODELS, "index_winners.npz")
IDX_ALL = os.path.join(MODELS, "index_all.npz")

YT_EMB = os.path.join(DATA, "clip_embeddings.npy")
YT_IDS = os.path.join(DATA, "clip_ids.npy")
PAID_EMB = os.path.join(DATA, "clip_embeddings_paid.npy")
PAID_IDS = os.path.join(DATA, "clip_ids_paid.npy")
PAID_META = os.path.join(DATA, "clip_paid_meta.csv")
THUMBS = os.path.join(DATA, "thumbnails")

# CV columns that exist in features.csv - these are what the LLM compares
CV_COLS = ["brightness", "contrast", "saturation", "colorfulness",
           "edge_density", "warm_ratio", "text_area_frac",
           "face_count", "face_area_frac"]
META_COLS = ["title_length", "title_word_count", "duration_seconds"]
SHOW_COLS = CV_COLS + META_COLS

try:
    import faiss  # noqa
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False


def norm_brand(x) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\.(com|in|co)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or "unknown"


def _thumb_map():
    if not os.path.isdir(THUMBS):
        return {}
    out = {}
    for f in os.listdir(THUMBS):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            out[os.path.splitext(f)[0]] = os.path.join("data", "thumbnails", f)
    return out


# ---------------------------------------------------------------- metadata
def build_youtube_meta(verbose=True) -> pd.DataFrame:
    sc = pd.read_csv(os.path.join(DATA, "youtube_scores.csv"))
    usecols = ["video_id", "channel_handle"] + CV_COLS + META_COLS
    feat = pd.read_csv(os.path.join(DATA, "features.csv"),
                       usecols=lambda c: c in usecols)
    df = sc.merge(feat, on="video_id", how="left")

    tp = os.path.join(DATA, "videos_labeled.csv")
    if os.path.exists(tp):
        t = pd.read_csv(tp, usecols=lambda c: c in ("video_id", "title"))
        t = t.drop_duplicates("video_id")
        df = df.merge(t, on="video_id", how="left")
    else:
        df["title"] = ""

    ch = os.path.join(DATA, "channels_resolved.csv")
    disp = {}
    if os.path.exists(ch):
        c = pd.read_csv(ch)
        for _, r in c.iterrows():
            if pd.notna(r.get("channel_title")):
                disp[norm_brand(r.get("handle"))] = str(r["channel_title"])

    thumbs = _thumb_map()
    out = pd.DataFrame({
        "key": df["video_id"].astype(str),
        "platform": "youtube",
        "brand": df["channel_handle"].map(norm_brand),
        "score": pd.to_numeric(df["score"], errors="coerce"),
        "label": pd.to_numeric(df.get("overperformed"), errors="coerce"),
        "labelable": df["labelable"].astype(str).str.lower().isin(["true", "1"]),
        "text": df.get("title", pd.Series("", index=df.index)).fillna("").astype(str),
        "md5": "",
    })
    out["display"] = out["brand"].map(lambda b: disp.get(b, b))
    out["path"] = out["key"].map(lambda k: thumbs.get(k, ""))
    for c in SHOW_COLS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    if verbose:
        print(f"  youtube meta: {len(out)} rows, "
              f"{out['brand'].nunique()} brands, "
              f"thumbs matched={int((out['path']!='').sum())}")
    return out


def build_paid_meta(verbose=True) -> pd.DataFrame:
    sc = pd.read_csv(os.path.join(DATA, "paid_ad_scores.csv"))
    df = pd.DataFrame({
        "key": sc["creative_id"].astype(str),
        "platform": sc["source"].astype(str).str.lower().str.strip(),
        "brand": sc["brand"].map(norm_brand),
        "score": pd.to_numeric(sc["score"], errors="coerce"),
        "label": np.nan,
        "labelable": False,
        "text": "",
    })
    df["display"] = sc["brand"].astype(str)

    op = os.path.join(DATA, "meta_ads_ocr.csv")
    if os.path.exists(op):
        o = pd.read_csv(op).drop_duplicates("creative_id")
        m = dict(zip(o["creative_id"].astype(str), o["ocr_text"].astype(str)))
        df["text"] = df["key"].map(lambda k: m.get(k, ""))

    if os.path.exists(PAID_META):
        pm = pd.read_csv(PAID_META).drop_duplicates("id")
        df["md5"] = df["key"].map(dict(zip(pm["id"].astype(str),
                                           pm.get("md5", pd.Series(dtype=str)).astype(str))))
        df["path"] = df["key"].map(dict(zip(pm["id"].astype(str),
                                            pm["path"].astype(str))))
    else:
        df["md5"], df["path"] = "", ""
    df["md5"] = df["md5"].fillna("")
    df["path"] = df["path"].fillna("")
    if verbose:
        print(f"  paid meta:    {len(df)} rows, "
              f"{df['platform'].value_counts().to_dict()}, "
              f"md5 present={int((df['md5']!='').sum())}")
    return df


# ---------------------------------------------------------------- build
def _load_pair(emb, ids, label):
    if not (os.path.exists(emb) and os.path.exists(ids)):
        print(f"  !! {label}: missing ({os.path.basename(emb)} / "
              f"{os.path.basename(ids)}) - skipped")
        return None, None
    E = np.load(emb).astype(np.float32)
    I = np.array([str(x) for x in np.load(ids, allow_pickle=True)])
    if len(E) != len(I):
        print(f"  !! {label}: {len(E)} vectors vs {len(I)} ids - skipped")
        return None, None
    print(f"  + {label:<10} {E.shape}")
    return E, I


def build():
    print("Building retrieval indexes...")
    print(f"  faiss: {HAVE_FAISS} (numpy path is used when filtering either way)")

    Ey, Iy = _load_pair(YT_EMB, YT_IDS, "youtube")
    Ep, Ip = _load_pair(PAID_EMB, PAID_IDS, "paid")
    if Ey is None and Ep is None:
        raise SystemExit("No embeddings. Run embed_paid.py first.")

    print()
    meta = pd.concat([d for d in (
        build_youtube_meta() if Ey is not None else None,
        build_paid_meta() if Ep is not None else None) if d is not None],
        ignore_index=True).drop_duplicates("key", keep="first").set_index("key")

    X = np.vstack([e for e in (Ey, Ep) if e is not None])
    keys = np.concatenate([i for i in (Iy, Ip) if i is not None])

    _, u = np.unique(keys, return_index=True)
    u.sort()
    X, keys = X[u], keys[u]

    hit = np.isin(keys, meta.index.values)
    print(f"\n  vectors={len(keys)}  matched to metadata={int(hit.sum())} "
          f"({100*hit.mean():.1f}%)")
    if hit.sum() < 0.5 * len(keys):
        print("  !! low match rate - id formats differ between .npy and CSV")
    X, keys = X[hit], keys[hit]
    md = meta.loc[keys].reset_index()

    # ---- md5 dedup. Without this, 61.6% of Google results are repeats.
    before = len(md)
    h = md["md5"].astype(str).str.strip()
    has = h.ne("") & h.str.lower().ne("nan")
    dup = has & md.assign(_h=h).duplicated(subset=["_h"], keep="first")
    md, X = md[~dup].reset_index(drop=True), X[(~dup).values]
    print(f"  md5 dedup: {before} -> {len(md)}  removed={before-len(md)} "
          f"(hashes present on {int(has.sum())} rows)")

    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    def save(path, mask, name):
        n = int(mask.sum())
        if n == 0:
            print(f"  !! {name}: 0 rows, not written")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sub = md[mask].reset_index(drop=True)
        np.savez_compressed(path, vectors=X[mask.values],
                            meta_json=json.dumps(sub.to_dict(orient="list"),
                                                 default=str))
        print(f"  wrote {os.path.relpath(path, ROOT):<28} rows={n:<7} "
              f"{os.path.getsize(path)/1e6:.1f} MB")

    is_yt = md["platform"] == "youtube"
    lab = md["labelable"].astype(str).str.lower().isin(["true", "1"])
    win = pd.to_numeric(md["label"], errors="coerce").fillna(0) >= 1
    save(IDX_WINNERS, is_yt & lab & win, "winners")
    save(IDX_ALL, pd.Series(True, index=md.index), "all")

    print("\n  index composition:")
    for pl, g in md.groupby("platform"):
        print(f"    {pl:<9} {len(g):<7} brands={g['brand'].nunique()}")
    print("\ndone.")


def stats():
    for p, n in ((IDX_WINNERS, "winners"), (IDX_ALL, "all")):
        if not os.path.exists(p):
            print(f"{n}: not built")
            continue
        ix = Index(p)
        print(f"\n[{n}] rows={len(ix.md)} dim={ix.V.shape[1]}")
        print(ix.md.groupby("platform")
              .agg(n=("key", "size"), brands=("brand", "nunique"),
                   mean_score=("score", "mean")).to_string())


# ---------------------------------------------------------------- search
class Hit:
    def __init__(self, r, sim):
        self.key = str(r["key"])
        self.sim = float(sim)
        self.brand = str(r.get("display") or r.get("brand", "?"))
        self.platform = str(r.get("platform", "?"))
        self.score = float(r["score"]) if pd.notna(r.get("score")) else float("nan")
        self.text = str(r.get("text", ""))[:140]
        self.path = str(r.get("path", ""))
        self.feats = {c: float(r[c]) for c in SHOW_COLS
                      if c in r.index and pd.notna(r.get(c))}


class Index:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.V = z["vectors"].astype(np.float32)
        self.md = pd.DataFrame(json.loads(str(z["meta_json"])))
        self.faiss = None
        if HAVE_FAISS:
            ix = faiss.IndexFlatIP(self.V.shape[1])
            ix.add(self.V)
            self.faiss = ix

    def query(self, v, k, mask=None):
        v = np.asarray(v, dtype=np.float32).ravel()
        v = v / (np.linalg.norm(v) + 1e-9)
        if mask is None and self.faiss is not None:
            s, i = self.faiss.search(v[None, :], min(k, len(self.V)))
            return i[0], s[0]
        sims = self.V @ v
        pool = np.where(mask)[0] if mask is not None else np.arange(len(sims))
        if pool.size == 0:
            return np.array([], int), np.array([])
        k = min(k, pool.size)
        top = pool[np.argpartition(-sims[pool], k - 1)[:k]]
        return top[np.argsort(-sims[top])], np.sort(sims[top])[::-1]


class Retriever:
    def __init__(self, winners=None, allidx=None):
        self.winners, self.all = winners, allidx

    @classmethod
    def load(cls):
        w = Index(IDX_WINNERS) if os.path.exists(IDX_WINNERS) else None
        a = Index(IDX_ALL) if os.path.exists(IDX_ALL) else None
        if w is None and a is None:
            raise FileNotFoundError("No indexes. Run: python src/retrieval.py --build")
        return cls(w, a)

    def search(self, vec, k=5, winners_only=True, brand=None, platform=None):
        """Filter preference: same brand -> same platform -> unrestricted.
        Falls back whenever the narrower pool has fewer than k rows."""
        ix = self.winners if (winners_only and self.winners) else self.all
        if ix is None:
            return []
        md, mask = ix.md, None
        if brand:
            bm = (md["brand"] == norm_brand(brand)).values
            if bm.sum() >= k:
                mask = bm
        if mask is None and platform:
            pm = (md["platform"] == platform).values
            if pm.sum() >= k:
                mask = pm
        idx, sims = ix.query(vec, k, mask)
        return [Hit(md.iloc[int(i)], s) for i, s in zip(idx, sims)]

    @staticmethod
    def prompt_block(hits, own_feats=None) -> str:
        """Deltas, not raw numbers. 'yours 0.31 vs median 0.14' is actionable;
        '0.31' alone is not."""
        if not hits:
            return "no comparable creatives retrieved"
        L = [f"{len(hits)} most visually similar high-performing creatives:"]
        for i, h in enumerate(hits, 1):
            f = " ".join(f"{k}={v:.3g}" for k, v in list(h.feats.items())[:6])
            t = f' text="{h.text}"' if h.text.strip() else ""
            L.append(f"  {i}. {h.brand}/{h.platform} sim={h.sim:.3f} "
                     f"score={h.score:.3f} {f}{t}")
        if own_feats:
            rows = []
            for k, v in own_feats.items():
                vals = [h.feats[k] for h in hits if k in h.feats]
                if not vals:
                    continue
                med = float(np.median(vals))
                rows.append(f"  {k}: yours={v:.3g} winners_median={med:.3g} "
                            f"delta={v-med:+.3g}")
            if rows:
                L.append("deltas vs retrieved winners:")
                L += rows
        return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.stats:
        stats()
    else:
        ap.print_help()
