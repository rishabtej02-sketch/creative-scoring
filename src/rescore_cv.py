#!/usr/bin/env python3
"""
rescore_cv.py - rebuild every stored score using the canonical CV extractor.

WHY:
score_youtube.py  imports cv_features     from app.py  (drifted)
score_paid_ads.py imports build_feature_row from app.py  (drifted)

Six of nine CV features in app.py disagreed with build_features.py, which
produced the training matrix. So data/youtube_scores.csv and
data/paid_ad_scores.csv were both computed with the wrong extractor. The
domain-transfer compression numbers derived from them must be recomputed.

Training and AUC 0.615 are unaffected - they were measured on features.csv.

WHAT THIS DOES NOT REDO:
CLIP. The stored embeddings are correct (embed_paid.py --verify reproduced
them at cosine 1.0000) and the bug was CV-only. So the CLIP model is never
loaded here: ~600 MB peak instead of ~1.8 GB, and no forward passes.

    data/clip_embeddings.npy      (13062, 512)  row i <-> clip_ids.npy[i]
    data/clip_embeddings_paid.npy (29405, 512)  row i <-> clip_paid_meta.csv row i

Old files are never overwritten. Outputs land at *_v2.csv so the report can
carry a before/after table.

Usage:
    python src/rescore_cv.py --which youtube
    python src/rescore_cv.py --which paid
    python src/rescore_cv.py --which both --resume
    python src/rescore_cv.py --compare          # stats only, no scoring
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

import scorer  # noqa: E402  - the canonical extractor
from scorer import CV_COLS, METADATA_COLS  # noqa: E402

YT_EMB = DATA / "clip_embeddings.npy"
YT_IDS = DATA / "clip_ids.npy"
YT_THUMBS = DATA / "thumbnails"
YT_OUT = DATA / "youtube_scores_v2.csv"
YT_OLD = DATA / "youtube_scores.csv"

PAID_EMB = DATA / "clip_embeddings_paid.npy"
PAID_LEDGER = DATA / "clip_paid_meta.csv"
PAID_OUT = DATA / "paid_ad_scores_v2.csv"
PAID_OLD = DATA / "paid_ad_scores.csv"

FLUSH_EVERY = 500


# ---------------------------------------------------------------- helpers
def load_bundle():
    """No CLIP. Model + tfidf + defaults + YuNet only."""
    b = scorer.load_all(with_clip=False, verbose=True)
    if b.face_det is None:
        print("  [warn] YuNet missing -> face features 0. Scores will NOT")
        print("         match training. Fix before trusting output.")
    return b


def make_row(meta: dict, cv: dict, clip_vec: np.ndarray, tfidf, title: str):
    """hstack([dense(METADATA_COLS + CV_COLS), clip_512, tfidf(title)])
    Strict training order. Same as scorer.build_feature_row, except the
    CLIP block is supplied instead of computed."""
    dense_vals = [meta[c] for c in METADATA_COLS] + [cv[c] for c in CV_COLS]
    dense = sparse.csr_matrix(
        np.array(dense_vals, dtype=np.float32).reshape(1, -1))
    clip_blk = sparse.csr_matrix(
        np.asarray(clip_vec, dtype=np.float32).reshape(1, -1))
    tfidf_blk = tfidf.transform([title]).astype(np.float32)
    return sparse.hstack([dense, clip_blk, tfidf_blk]).tocsr()


def done_ids(path: Path, col: str) -> set:
    if not path.exists():
        return set()
    try:
        d = pd.read_csv(path, encoding="utf-8", usecols=[col])
        return set(d[col].astype(str))
    except Exception:
        return set()


def open_out(path: Path, header, resume: bool):
    mode = "a" if (resume and path.exists()) else "w"
    f = path.open(mode, newline="", encoding="utf-8")
    w = csv.writer(f)
    if mode == "w":
        w.writerow(header)
    return f, w


def stats(v):
    v = np.asarray(v, dtype=float)
    return dict(n=len(v), mean=float(v.mean()), std=float(v.std()),
                p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)))


# ---------------------------------------------------------------- youtube
def rescore_youtube(bundle, resume=False, limit=None):
    emb = np.load(YT_EMB)
    ids = np.load(YT_IDS, allow_pickle=True)
    assert len(emb) == len(ids), f"emb {len(emb)} != ids {len(ids)}"
    id2row = {str(v): i for i, v in enumerate(ids)}
    print(f"youtube: {len(ids)} stored CLIP vectors")

    df = pd.read_csv(DATA / "features.csv", encoding="utf-8")
    df = df[df["video_id"].astype(str).isin(id2row)].reset_index(drop=True)
    print(f"youtube: {len(df)} rows in features.csv with a CLIP vector")

    # titles - same lookup order score_youtube.py used
    title_map = {}
    for cand in ("videos_labeled.csv", "videos_raw.csv"):
        p = DATA / cand
        if not p.exists():
            continue
        t = pd.read_csv(p, encoding="utf-8",
                        usecols=lambda c: c in ("video_id", "title"))
        if "title" in t.columns:
            title_map = dict(zip(t["video_id"].astype(str),
                                 t["title"].fillna("")))
            print(f"youtube: titles from {cand} -> {len(title_map)}")
            break
    if not title_map:
        print("youtube: [warn] no title source -> title=''")

    skip = done_ids(YT_OUT, "video_id") if resume else set()
    if skip:
        print(f"youtube: resume, {len(skip)} already scored")
    f, w = open_out(YT_OUT, ["video_id", "score", "overperformed",
                             "labelable"], resume)

    have_lbl = "overperformed" in df.columns
    have_able = "labelable" in df.columns
    buf, n, fail = [], 0, 0
    try:
        for _, r in tqdm(df.iterrows(), total=len(df), desc="yt "):
            vid = str(r["video_id"])
            if vid in skip:
                continue
            p = YT_THUMBS / f"{vid}.jpg"
            if not p.exists():
                fail += 1
                continue
            try:
                img = scorer.load_bgr(p)
                cv = scorer.cv_features(img, bundle.face_det)
                H, W = img.shape[:2]
                title = title_map.get(vid, "")
                meta = {}
                for c in METADATA_COLS:
                    if c in df.columns and pd.notna(r.get(c)):
                        meta[c] = float(r[c])
                    elif c == "title_length":
                        meta[c] = float(len(title))
                    elif c == "title_word_count":
                        meta[c] = float(len(title.split()))
                    elif c == "is_short":
                        meta[c] = 1.0 if H > W else 0.0
                    else:
                        meta[c] = float(bundle.defaults.get(c, 0.0))
                row = make_row(meta, cv, emb[id2row[vid]], bundle.tfidf, title)
                s = float(bundle.model.predict_proba(row)[:, 1][0])
            except Exception:
                fail += 1
                continue
            buf.append([vid, s,
                        r["overperformed"] if have_lbl else "",
                        r["labelable"] if have_able else ""])
            n += 1
            if len(buf) >= FLUSH_EVERY:
                w.writerows(buf)
                f.flush()
                buf = []
            if limit and n >= limit:
                break
    finally:
        if buf:
            w.writerows(buf)
        f.close()
    print(f"youtube: scored {n}, failed/missing {fail} -> {YT_OUT}")


# ------------------------------------------------------------------- paid
def rescore_paid(bundle, resume=False, limit=None):
    emb = np.load(PAID_EMB)
    led = pd.read_csv(PAID_LEDGER, encoding="utf-8")
    assert len(emb) == len(led), f"emb {len(emb)} != ledger {len(led)}"
    print(f"paid: {len(led)} stored CLIP vectors, row-aligned to ledger")

    skip = done_ids(PAID_OUT, "creative_id") if resume else set()
    if skip:
        print(f"paid: resume, {len(skip)} already scored")
    f, w = open_out(PAID_OUT, ["source", "brand", "creative_id", "score"],
                    resume)

    # title = "" for paid creatives. ad_text is legal disclosure, not signal.
    tfidf_blank = bundle.tfidf.transform([""]).astype(np.float32)

    buf, n, fail = [], 0, 0
    try:
        for i, r in tqdm(led.iterrows(), total=len(led), desc="paid"):
            cid = str(r["id"])
            if cid in skip:
                continue
            rel = str(r["path"]).replace("\\", "/")
            p = ROOT / rel
            if not p.exists():
                p = DATA / Path(rel).name
            if not p.exists():
                fail += 1
                continue
            try:
                img = scorer.load_bgr(p)
                cv = scorer.cv_features(img, bundle.face_det)
                H, W = img.shape[:2]
                meta = {}
                for c in METADATA_COLS:
                    if c == "title_length" or c == "title_word_count":
                        meta[c] = 0.0
                    elif c == "is_short":
                        meta[c] = 1.0 if H > W else 0.0
                    else:
                        meta[c] = float(bundle.defaults.get(c, 0.0))
                dense_vals = ([meta[c] for c in METADATA_COLS]
                              + [cv[c] for c in CV_COLS])
                dense = sparse.csr_matrix(
                    np.array(dense_vals, dtype=np.float32).reshape(1, -1))
                clip_blk = sparse.csr_matrix(
                    emb[i].astype(np.float32).reshape(1, -1))
                row = sparse.hstack([dense, clip_blk, tfidf_blank]).tocsr()
                s = float(bundle.model.predict_proba(row)[:, 1][0])
            except Exception:
                fail += 1
                continue
            brand = Path(rel).parent.name
            buf.append([str(r["platform"]), brand, cid, s])
            n += 1
            if len(buf) >= FLUSH_EVERY:
                w.writerows(buf)
                f.flush()
                buf = []
            if limit and n >= limit:
                break
    finally:
        if buf:
            w.writerows(buf)
        f.close()
    print(f"paid: scored {n}, failed/missing {fail} -> {PAID_OUT}")


# ---------------------------------------------------------------- compare
def compare():
    print("\n=== OLD vs NEW ===")
    rows = []
    if YT_OLD.exists():
        rows.append(("youtube OLD", pd.read_csv(YT_OLD)["score"]))
    if YT_OUT.exists():
        rows.append(("youtube NEW", pd.read_csv(YT_OUT)["score"]))
    for tag, path in (("OLD", PAID_OLD), ("NEW", PAID_OUT)):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        for plat in sorted(d["source"].unique()):
            rows.append((f"{plat} {tag}", d.loc[d["source"] == plat, "score"]))

    print(f"{'set':<16}{'n':>7}{'mean':>9}{'std':>9}{'p10':>9}{'p90':>9}")
    print("-" * 59)
    keep = {}
    for name, v in rows:
        s = stats(v)
        keep[name] = s
        print(f"{name:<16}{s['n']:>7}{s['mean']:>9.4f}{s['std']:>9.4f}"
              f"{s['p10']:>9.4f}{s['p90']:>9.4f}")

    for tag in ("OLD", "NEW"):
        yt = keep.get(f"youtube {tag.lower()}") or keep.get(f"youtube {tag}")
        if not yt:
            continue
        print(f"\ncompression vs youtube ({tag}), sigma ratio:")
        for plat in ("meta", "google"):
            k = f"{plat} {tag}"
            if k in keep and keep[k]["std"] > 0:
                print(f"  {plat:<8} {yt['std'] / keep[k]['std']:.2f}x")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["youtube", "paid", "both"],
                    default="both")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping ids already in the _v2 file")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N images (smoke test)")
    ap.add_argument("--compare", action="store_true",
                    help="print old vs new stats only, no scoring")
    a = ap.parse_args()

    if a.compare:
        compare()
        return

    b = load_bundle()
    if a.which in ("youtube", "both"):
        rescore_youtube(b, a.resume, a.limit)
    if a.which in ("paid", "both"):
        rescore_paid(b, a.resume, a.limit)
    compare()


if __name__ == "__main__":
    main()
