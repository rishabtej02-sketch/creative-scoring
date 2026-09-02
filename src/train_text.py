"""
Step 3b: Add TITLE TEXT as features via TF-IDF, and measure what it adds.

TF-IDF turns each title into numbers: a word scores high when it is frequent
in THIS title but rare across all titles (so "sale" in a sea of generic words
carries weight; "the" carries none). This is the classic text baseline — and
your plan is explicit that any GenAI text feature must later be benchmarked
against exactly this, so we build it now as the thing to beat.

IMPORTANT — no leakage: TF-IDF is fit on the TRAIN titles only, then applied
to the test titles. Fitting on all titles would let test-set vocabulary bleed
into training.

Two experiments, same time-split and model as before:
  1. text_only              — can the words alone predict overperformance?
  2. metadata_cv_text       — does text ADD anything on top of everything else?

Reads  data/features.csv, data/videos_raw.csv (for titles)
Appends to data/model_results.csv

Usage:
  python train_text.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "features.csv"
RAW_CSV = DATA_DIR / "videos_raw.csv"
OUT_CSV = DATA_DIR / "model_results.csv"

TEST_FRACTION = 0.2
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 300   # keep the top-N words; avoids a huge sparse blowup

# Dense numeric feature columns reused from the earlier trainer.
METADATA_COLS = [
    "duration_seconds", "hour_of_day", "day_of_week", "is_weekend",
    "title_length", "title_word_count", "tag_count", "description_length",
    "category_id", "has_caption", "is_short",
]
CV_COLS = [
    "brightness", "contrast", "saturation", "colorfulness",
    "edge_density", "warm_ratio", "text_area_frac",
    "face_count", "face_area_frac",
]


def load():
    if not IN_CSV.exists():
        sys.exit(f"Not found: {IN_CSV}. Run build_features.py first.")
    if not RAW_CSV.exists():
        sys.exit(f"Not found: {RAW_CSV}. Titles come from the raw file.")

    df = pd.read_csv(IN_CSV, encoding="utf-8")
    titles = pd.read_csv(RAW_CSV, encoding="utf-8", usecols=["video_id", "title"])
    df = df.merge(titles, on="video_id", how="left")
    df["title"] = df["title"].fillna("").astype(str)

    df = df[df["labelable"] == True].copy()  # noqa: E712
    df["overperformed"] = pd.to_numeric(df["overperformed"], errors="coerce")
    df = df[df["overperformed"].notna()]
    df["overperformed"] = df["overperformed"].astype(int)
    df["is_short"] = (df["format"] == "short").astype(int)

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at").reset_index(drop=True)
    print(f"Loaded {len(df):,} labeled rows with titles")
    return df


def time_split_idx(n):
    cut = int(n * (1 - TEST_FRACTION))
    return cut


def fit_tfidf(train_titles, test_titles):
    """Fit on TRAIN titles only, transform both. Returns sparse matrices + vocab size."""
    vec = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        stop_words="english",     # drop the/a/is/...
        ngram_range=(1, 2),       # single words + pairs ("free trial")
        min_df=5,                 # ignore words in <5 titles (typos, noise)
    )
    Xtr = vec.fit_transform(train_titles)
    Xte = vec.transform(test_titles)
    return Xtr, Xte, len(vec.vocabulary_)


def evaluate(name, Xtr, ytr, Xte, yte, n_feat):
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    acc = accuracy_score(yte, (proba >= 0.5).astype(int))
    print(f"  {name:24s} AUC {auc:.3f} | acc {acc:.3f} | n_feat {n_feat} | test {len(yte):,}")
    return {"experiment": name, "auc": round(auc, 4), "accuracy": round(acc, 4),
            "n_features": n_feat, "n_train": Xtr.shape[0], "n_test": len(yte)}


def main():
    df = load()
    y = df["overperformed"].values
    cut = time_split_idx(len(df))
    ytr, yte = y[:cut], y[cut:]
    print(f"  train {cut:,} (older) | test {len(df) - cut:,} (newer)")
    print(f"  split at {df['published_at'].iloc[cut].date()}\n")

    # --- TF-IDF matrices (train-fit only) ---
    Xtr_txt, Xte_txt, vocab = fit_tfidf(
        df["title"].iloc[:cut], df["title"].iloc[cut:]
    )
    print(f"TF-IDF vocab kept: {vocab} terms\n")

    results = []

    # 1. text only
    results.append(evaluate("text_only", Xtr_txt, ytr, Xte_txt, yte, vocab))

    # 2. metadata + cv + text (dense numeric stacked with sparse text)
    dense = df[METADATA_COLS + CV_COLS].apply(pd.to_numeric, errors="coerce").fillna(-1)
    dense_tr = csr_matrix(dense.iloc[:cut].values)
    dense_te = csr_matrix(dense.iloc[cut:].values)
    Xtr_all = hstack([dense_tr, Xtr_txt]).tocsr()
    Xte_all = hstack([dense_te, Xte_txt]).tocsr()
    nfeat_all = dense.shape[1] + vocab
    results.append(evaluate("metadata_cv_text", Xtr_all, ytr, Xte_all, yte, nfeat_all))

    # append to existing results file if present
    new = pd.DataFrame(results)
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nAppended {len(new)} rows -> {OUT_CSV}")

    print("\n" + "=" * 58)
    print("Compare metadata_cv_text here vs metadata_plus_cv earlier.")
    print("Rise = titles carry signal. Flat = words don't help.")
    print("text_only tells you how much the words alone know.")


if __name__ == "__main__":
    main()
