"""
Step 3: Train and compare models on a TIME-BASED split.

The split is chronological, not random: we train on the OLDER videos and
test on the NEWER ones. This mirrors the real task — stand at upload time,
predict whether the creative will beat its channel's track record — and it
prevents leakage that a random split would allow (a channel's future video
helping predict its past one).

We fit the SAME model (LightGBM) on three feature sets so the comparison
is apples-to-apples, and the GAPS between them are the finding:

  1. metadata-only   — no pixels at all. The "does timing/format predict
                       performance better than the creative?" baseline.
  2. cv-only         — thumbnail features alone.
  3. metadata + cv   — everything, with is_short included.

Then we refit set (3) SEPARATELY on shorts and on longs, because the two
formats may have different drivers — and if they do, that is itself a
result worth reporting.

Metric: ROC AUC (probability the model ranks a real winner above a real
loser) plus accuracy. AUC is threshold-free and unfooled by class balance.

Reads  data/features.csv
Writes data/model_results.csv  (one row per experiment)

Usage:
  pip install lightgbm scikit-learn pandas numpy
  python train_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "features.csv"
RAW_CSV = DATA_DIR / "videos_raw.csv"   # source of extra metadata not in features.csv
OUT_CSV = DATA_DIR / "model_results.csv"

TEST_FRACTION = 0.2       # newest 20% of videos become the test set
RANDOM_STATE = 42
RECENT_YEARS = 3          # for the "recent only" comparison experiment

# Columns that are NOT features (ids, target, bookkeeping).
NON_FEATURE = {
    "video_id", "channel_handle", "channel_id", "format",
    "published_at", "view_count", "baseline_median", "prior_count",
    "labelable", "overperformed", "has_thumbnail",
}

# Extra metadata pulled from videos_raw.csv. ALL known at upload time.
# NOTE: like_count / comment_count are DELIBERATELY excluded — they are
# post-publication outcomes, so using them would leak the future, exactly
# like using view_count would.
EXTRA_META = ["default_audio_language", "definition", "licensed_content"]

# Categorical features: LightGBM must treat these as unordered labels, not
# numbers. category_id 24 is not "bigger" than 22. Encoded as pandas
# 'category' dtype, which LightGBM reads natively.
CATEGORICAL = ["category_id", "default_audio_language", "definition",
               "licensed_content", "day_of_week", "hour_of_day"]

# Feature groups. is_short is derived below and added to metadata.
METADATA_COLS = [
    "duration_seconds", "hour_of_day", "day_of_week", "is_weekend",
    "title_length", "title_word_count", "tag_count", "description_length",
    "category_id", "has_caption", "is_short",
    "default_audio_language", "definition", "licensed_content",
]
CV_COLS = [
    "brightness", "contrast", "saturation", "colorfulness",
    "edge_density", "warm_ratio", "text_area_frac",
    "face_count", "face_area_frac",
]


def load():
    if not IN_CSV.exists():
        sys.exit(f"Not found: {IN_CSV}. Run build_features.py first.")
    df = pd.read_csv(IN_CSV, encoding="utf-8")

    # Merge in extra upload-time metadata from the raw collection file,
    # matched by video_id. Avoids re-running the slow CV step.
    if RAW_CSV.exists():
        raw = pd.read_csv(RAW_CSV, encoding="utf-8",
                          usecols=["video_id"] + EXTRA_META)
        df = df.merge(raw, on="video_id", how="left")
        print(f"Merged extra metadata: {EXTRA_META}")
    else:
        print(f"WARNING: {RAW_CSV} not found — extra metadata skipped")
        for c in EXTRA_META:
            df[c] = np.nan

    # keep only rows that actually carry a label
    df = df[df["labelable"] == True].copy()  # noqa: E712
    df["overperformed"] = pd.to_numeric(df["overperformed"], errors="coerce")
    df = df[df["overperformed"].notna()]
    df["overperformed"] = df["overperformed"].astype(int)

    # derive is_short from the format column we carried through
    df["is_short"] = (df["format"] == "short").astype(int)

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at").reset_index(drop=True)

    # Set categorical dtype so LightGBM treats these as unordered labels.
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")

    print(f"Loaded {len(df):,} labeled rows")
    print(f"  win rate: {df['overperformed'].mean():.1%}")
    print(f"  date range: {df['published_at'].min().date()} -> "
          f"{df['published_at'].max().date()}")
    return df


def time_split(df):
    """Oldest (1-TEST_FRACTION) train, newest TEST_FRACTION test."""
    cut = int(len(df) * (1 - TEST_FRACTION))
    train, test = df.iloc[:cut], df.iloc[cut:]
    split_date = test["published_at"].min().date()
    print(f"  train {len(train):,} (older)  |  test {len(test):,} (newer)")
    print(f"  split at {split_date}")
    return train, test


def run_experiment(name, cols, train, test):
    Xtr, ytr = train[cols], train["overperformed"]
    Xte, yte = test[cols], test["overperformed"]

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(Xtr, ytr, categorical_feature="auto")

    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    acc = accuracy_score(yte, (proba >= 0.5).astype(int))

    print(f"  {name:28s} AUC {auc:.3f} | acc {acc:.3f} | "
          f"n_feat {len(cols):2d} | test {len(yte):,}")
    return {"experiment": name, "auc": round(auc, 4),
            "accuracy": round(acc, 4), "n_features": len(cols),
            "n_train": len(ytr), "n_test": len(yte)}, model


def top_features(model, cols, k=8):
    imp = sorted(zip(cols, model.feature_importances_),
                 key=lambda t: t[1], reverse=True)
    return ", ".join(f"{c}({v})" for c, v in imp[:k])


def main():
    df = load()
    results = []

    print("\n--- Whole dataset, time-split ---")
    train, test = time_split(df)

    r1, _ = run_experiment("metadata_only", METADATA_COLS, train, test)
    r2, _ = run_experiment("cv_only", CV_COLS, train, test)
    r3, m3 = run_experiment("metadata_plus_cv", METADATA_COLS + CV_COLS, train, test)
    results += [r1, r2, r3]

    print("\n  Top features (metadata+cv):")
    print("   ", top_features(m3, METADATA_COLS + CV_COLS))

    # ---- per-format split: refit the full feature set separately ----
    print("\n--- Per-format (metadata+cv), each time-split on its own ---")
    for fmt in ("short", "long"):
        sub = df[df["format"] == fmt].reset_index(drop=True)
        if len(sub) < 200:
            print(f"  {fmt}: only {len(sub)} rows, skipping")
            continue
        tr, te = time_split(sub)
        cols = [c for c in METADATA_COLS + CV_COLS if c != "is_short"]
        r, _ = run_experiment(f"{fmt}_only_metadata_plus_cv", cols, tr, te)
        results.append(r)

    # ---- recency test: does cutting old data help? (your era-drift theory) ----
    print(f"\n--- Recent {RECENT_YEARS} years only (metadata+cv) ---")
    latest = df["published_at"].max()
    cutoff = latest - pd.DateOffset(years=RECENT_YEARS)
    recent = df[df["published_at"] >= cutoff].reset_index(drop=True)
    print(f"  {len(recent):,} rows since {cutoff.date()} "
          f"(vs {len(df):,} all-time)")
    if len(recent) >= 400:
        tr, te = time_split(recent)
        r, _ = run_experiment("recent_metadata_plus_cv",
                              METADATA_COLS + CV_COLS, tr, te)
        results.append(r)
        print(f"\n  Compare: all-time metadata+cv AUC {r3['auc']} "
              f"vs recent AUC {r['auc']}")
    else:
        print("  too few recent rows, skipping")

    out = pd.DataFrame(results)
    DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nWrote results -> {OUT_CSV}")

    print("\n" + "=" * 60)
    print("READING THE RESULTS")
    print("=" * 60)
    print("- 0.50 AUC = coin flip. ~0.62-0.68 expected here.")
    print("- metadata_only vs cv_only: does the image beat pure format/timing?")
    print("- metadata_plus_cv minus metadata_only: what do pixels ADD?")
    print("- short vs long: do the two formats have different drivers?")


if __name__ == "__main__":
    main()
