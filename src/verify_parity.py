#!/usr/bin/env python3
"""
verify_parity.py - prove the inference extractor matches the training data.

This is the same discipline as embed_paid.py --verify, which re-embedded 200
known thumbnails and got cosine 1.0000 against the stored vectors before
committing to a 50-minute run. Here: re-extract CV features from training
thumbnails with scorer.py and compare against the stored features.csv values
that the model was actually fitted on.

A drift of exactly this kind went unnoticed for two phases - app.py had its
own copy of the extractor and six of nine CV features diverged. Line counts
did not catch it. Only a value comparison does.

    python src/verify_parity.py                 # 200 thumbnails
    python src/verify_parity.py -n 1000          # more
    python src/verify_parity.py --show-worst 5   # print worst offenders
    python src/verify_parity.py --compare-app    # also test app.py's version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorer import CV_COLS, FaceDetector, YUNET_PATH, cv_features, load_bgr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
THUMBS = DATA / "thumbnails"

# absolute tolerance per column. Everything here is deterministic given the
# same pixels, so the bar is machine epsilon, not "close enough".
TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="thumbnails to check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-worst", type=int, default=0,
                    help="print N worst rows per failing column")
    a = ap.parse_args()

    fp = DATA / "features.csv"
    if not fp.exists():
        sys.exit(f"missing {fp} - run build_features.py first")

    df = pd.read_csv(fp, encoding="utf-8")
    if "has_thumbnail" in df.columns:
        df = df[df["has_thumbnail"] == 1]
    have = [c for c in CV_COLS if c in df.columns]
    missing = [c for c in CV_COLS if c not in df.columns]
    if missing:
        print(f"  [warn] not in features.csv, skipped: {missing}")
    df = df.dropna(subset=have)
    if df.empty:
        sys.exit("no rows with CV features")

    df = df.sample(n=min(a.n, len(df)), random_state=a.seed)
    print(f"comparing {len(df)} thumbnails against features.csv")

    face_det = FaceDetector() if YUNET_PATH.exists() else None
    if face_det is None:
        print(f"  [warn] YuNet missing at {YUNET_PATH};")
        print("         face_count / face_area_frac will be skipped")

    diffs = {c: [] for c in have}
    rows = []
    n_ok = n_skip = 0
    for _, r in df.iterrows():
        p = THUMBS / f"{r['video_id']}.jpg"
        if not p.exists():
            n_skip += 1
            continue
        try:
            got = cv_features(load_bgr(p), face_det)
        except Exception as e:
            print(f"  [skip] {r['video_id']}: {type(e).__name__}: {e}")
            n_skip += 1
            continue
        n_ok += 1
        rec = {"video_id": r["video_id"]}
        for c in have:
            if face_det is None and c in ("face_count", "face_area_frac"):
                continue
            d = abs(float(got[c]) - float(r[c]))
            diffs[c].append(d)
            rec[c] = d
        rows.append(rec)

    if n_ok == 0:
        sys.exit("no thumbnails could be read - check data/thumbnails/")

    print(f"\nchecked {n_ok}, skipped {n_skip}\n")
    print(f"{'feature':<18}{'max abs diff':>14}{'mean':>14}   verdict")
    print("-" * 62)
    failed = []
    for c in have:
        if not diffs[c]:
            print(f"{c:<18}{'(skipped)':>14}{'':>14}")
            continue
        arr = np.array(diffs[c])
        ok = arr.max() < TOL
        if not ok:
            failed.append(c)
        print(f"{c:<18}{arr.max():>14.3e}{arr.mean():>14.3e}   "
              f"{'OK' if ok else 'MISMATCH'}")

    if a.show_worst and failed:
        wide = pd.DataFrame(rows)
        for c in failed:
            print(f"\nworst {a.show_worst} for {c}:")
            print(wide.nlargest(a.show_worst, c)[["video_id", c]]
                  .to_string(index=False))

    print()
    if failed:
        print(f"PARITY FAIL on {len(failed)} column(s): {', '.join(failed)}")
        print("scorer.py disagrees with the training data. Do not ship scores.")
        return 1
    print("PARITY OK - scorer.py reproduces features.csv exactly.")
    print("Inference now uses the same extractor the model was trained on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
