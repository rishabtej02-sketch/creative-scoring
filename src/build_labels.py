"""
Step 2a: Build the prediction label from collected video metadata.

The label is binary within-channel-within-format overperformance:
"did this video beat the TRAILING median view count of the same channel's
PAST videos of the same format (short vs long)?"

Three rules make this defensible, and each maps to a report sentence:

  1. TRAILING, not all-time. A video is compared only to videos of the
     same channel published strictly BEFORE it. Using future videos to
     set the baseline would leak information that did not exist at upload
     time — and this project predicts upload-time. Past predicts future.

  2. WITHIN FORMAT. Shorts are compared only to past shorts, long-form
     only to past long-form. Shorts pull far more views, so a mixed
     baseline would make the label mostly "is this a short?" — that is
     format, not creative quality. Split by duration (<=60s = short).

  3. MIN PRIOR = 10. A median over fewer than 10 prior videos is noisy,
     so a video with fewer than 10 same-channel same-format predecessors
     gets NO label. Those early videos are NOT discarded — they still
     count as history for later videos. They just cannot be scored fairly.

Also filtered out before labeling:
  - live / upcoming broadcasts
  - rows with no view_count (channel hid statistics)
  - videos younger than MATURITY_DAYS (views have not settled yet)

Usage:
    python build_labels.py
    # reads  data/videos_raw.csv
    # writes data/videos_labeled.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "videos_raw.csv"
OUT_CSV = DATA_DIR / "videos_labeled.csv"

MATURITY_DAYS = 90       # views must have had time to settle
SHORT_MAX_SECONDS = 60   # <=60s counts as short-form
MIN_PRIOR = 10           # need this many past same-format videos to label
NOW = pd.Timestamp.now(tz="UTC")


def load(path):
    if not path.exists():
        sys.exit(f"Not found: {path}. Run the collector first.")
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def clean(df):
    """Drop junk rows. Print what goes and why."""
    start = len(df)

    # published_at -> real datetime; unparseable rows dropped
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df[df["published_at"].notna()]
    print(f"  dropped {start - len(df):,} rows: bad/missing publish date")

    # live / upcoming
    before = len(df)
    df = df[df["live_broadcast_content"].fillna("none") == "none"]
    print(f"  dropped {before - len(df):,} rows: live / upcoming")

    # view_count must be a real number
    before = len(df)
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce")
    df = df[df["view_count"].notna()]
    print(f"  dropped {before - len(df):,} rows: missing view_count")

    # duration must exist to decide short vs long
    before = len(df)
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    df = df[df["duration_seconds"].notna()]
    print(f"  dropped {before - len(df):,} rows: missing duration")

    # maturity: old enough that views have settled
    before = len(df)
    age_days = (NOW - df["published_at"]).dt.total_seconds() / 86400
    df = df[age_days >= MATURITY_DAYS]
    print(f"  dropped {before - len(df):,} rows: younger than {MATURITY_DAYS} days")

    print(f"  {len(df):,} rows survive cleaning")
    return df


def add_format(df):
    df = df.copy()
    df["format"] = np.where(
        df["duration_seconds"] <= SHORT_MAX_SECONDS, "short", "long"
    )
    return df


def label_group(g):
    """
    One (channel, format) group, in time order.
    For each video, baseline = median view_count of the STRICTLY earlier
    videos in this same group. Need >= MIN_PRIOR of them, else no label.

    Leak-safe by construction: .shift() drops the current row out of every
    window, so a video never sees itself or any later video.
    """
    g = g.sort_values("published_at").copy()
    views = g["view_count"]

    # expanding().median() at row i covers rows 0..i (includes self);
    # shift(1) turns it into "median of rows 0..i-1" (past only).
    trailing_median = views.expanding().median().shift(1)
    prior_count = np.arange(len(g))  # rows strictly before each row

    g["baseline_median"] = trailing_median
    g["prior_count"] = prior_count
    g["labelable"] = prior_count >= MIN_PRIOR
    g["overperformed"] = np.where(
        g["labelable"], (views > trailing_median).astype("Int64"), pd.NA
    )
    return g


def build_labels(df):
    df = add_format(df)
    pieces = []
    for _, g in df.groupby(["channel_handle", "format"], sort=False):
        pieces.append(label_group(g))
    return pd.concat(pieces, ignore_index=True)


def summarise(labeled):
    # groupby-apply can drop the grouping column from the result; recompute it
    if "format" not in labeled.columns:
        labeled = labeled.copy()
        labeled["format"] = np.where(
            labeled["duration_seconds"] <= SHORT_MAX_SECONDS, "short", "long"
        )
    labeled_only = labeled[labeled["labelable"]]
    print("\n" + "=" * 60)
    print("LABEL SUMMARY")
    print("=" * 60)
    print(f"Rows after cleaning        : {len(labeled):,}")
    print(f"Rows with a label          : {len(labeled_only):,}")
    print(f"Rows unlabeled (early)     : {len(labeled) - len(labeled_only):,}")

    if len(labeled_only):
        wins = int(labeled_only["overperformed"].sum())
        pct = 100 * wins / len(labeled_only)
        print(f"Class balance (win=1)      : {wins:,} / {len(labeled_only):,} ({pct:.1f}%)")

    by_fmt = labeled_only.groupby("format").size()
    print("\nLabeled rows by format:")
    for fmt, n in by_fmt.items():
        print(f"   {fmt:5s}: {n:,}")

    per_ch = labeled_only.groupby("channel_handle").size().sort_values()
    thin = per_ch[per_ch < 40]
    print(f"\nChannels with <40 labeled rows ({len(thin)}):")
    for ch, n in thin.items():
        print(f"   {ch}: {n}")
    print(f"\nChannels with >=40 labeled rows: {(per_ch >= 40).sum()}")


def main():
    df = load(IN_CSV)
    df = clean(df)
    labeled = build_labels(df)

    # ensure format column is present in the written file
    if "format" not in labeled.columns:
        labeled["format"] = np.where(
            labeled["duration_seconds"] <= SHORT_MAX_SECONDS, "short", "long"
        )

    DATA_DIR.mkdir(exist_ok=True)
    labeled.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {len(labeled):,} rows -> {OUT_CSV}")
    print("(only rows where labelable=True carry a win/lose label)")

    summarise(labeled)


if __name__ == "__main__":
    main()
