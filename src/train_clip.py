"""
Step 3c (part 2): Train on the cached CLIP embeddings and compare.

Loads the 512-d CLIP vectors produced by embed_clip.py, lines them up with
the labels by video_id, and runs the same time-split gradient-boosting
comparison as before:

  1. clip_only            — do semantic image vectors alone predict?
  2. clip_plus_metadata_cv_text — does CLIP ADD on top of everything so far?

The comparison that matters: clip_only vs cv_only (handcrafted). If CLIP
beats handcrafted, deep semantic content carries signal the simple stats
missed. If it lands at the same ~0.58 ceiling, that is a strong result too:
even deep image meaning does not crack the problem once brand size is
normalized out.

Reads  data/features.csv, data/videos_raw.csv (titles),
       data/clip_embeddings.npy, data/clip_ids.npy
Appends to data/model_results.csv

Usage:
  python train_clip.py
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
EMB_PATH = DATA_DIR / "clip_embeddings.npy"
IDS_PATH = DATA_DIR / "clip_ids.npy"
OUT_CSV = DATA_DIR / "model_results.csv"

TEST_FRACTION = 0.2
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 300

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
    for p in (IN_CSV, RAW_CSV, EMB_PATH, IDS_PATH):
        if not p.exists():
            sys.exit(f"Not found: {p}. Run build_features.py, train_text setup, "
                     f"and embed_clip.py first.")

    df = pd.read_csv(IN_CSV, encoding="utf-8")
    titles = pd.read_csv(RAW_CSV, encoding="utf-8", usecols=["video_id", "title"])
    df = df.merge(titles, on="video_id", how="left")
    df["title"] = df["title"].fillna("").astype(str)

    # attach CLIP embeddings by video_id
    emb = np.load(EMB_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True)
    emb_map = {vid: emb[i] for i, vid in enumerate(ids)}
    has_emb = df["video_id"].isin(emb_map)
    print(f"Rows with a CLIP embedding: {has_emb.sum():,} / {len(df):,}")
    df = df[has_emb].copy()

    df = df[df["labelable"] == True].copy()  # noqa: E712
    df["overperformed"] = pd.to_numeric(df["overperformed"], errors="coerce")
    df = df[df["overperformed"].notna()]
    df["overperformed"] = df["overperformed"].astype(int)
    df["is_short"] = (df["format"] == "short").astype(int)

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at").reset_index(drop=True)

    clip_mat = np.vstack([emb_map[v] for v in df["video_id"]]).astype("float32")
    print(f"Loaded {len(df):,} labeled rows with embeddings ({clip_mat.shape[1]} dims)")
    return df, clip_mat


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
    print(f"  {name:32s} AUC {auc:.3f} | acc {acc:.3f} | n_feat {n_feat} | test {len(yte):,}")
    return {"experiment": name, "auc": round(auc, 4), "accuracy": round(acc, 4),
            "n_features": n_feat, "n_train": Xtr.shape[0], "n_test": len(yte)}


def main():
    df, clip_mat = load()
    y = df["overperformed"].values
    cut = int(len(df) * (1 - TEST_FRACTION))
    ytr, yte = y[:cut], y[cut:]
    print(f"  train {cut:,} (older) | test {len(df) - cut:,} (newer)")
    print(f"  split at {df['published_at'].iloc[cut].date()}\n")

    results = []

    # 1. CLIP only
    results.append(evaluate("clip_only",
                            clip_mat[:cut], ytr, clip_mat[cut:], yte,
                            clip_mat.shape[1]))

    # 2. CLIP + metadata + cv + text (everything)
    dense = df[METADATA_COLS + CV_COLS].apply(pd.to_numeric, errors="coerce").fillna(-1)
    vec = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, stop_words="english",
                          ngram_range=(1, 2), min_df=5)
    txt_tr = vec.fit_transform(df["title"].iloc[:cut])
    txt_te = vec.transform(df["title"].iloc[cut:])

    Xtr = hstack([csr_matrix(dense.iloc[:cut].values),
                  csr_matrix(clip_mat[:cut]), txt_tr]).tocsr()
    Xte = hstack([csr_matrix(dense.iloc[cut:].values),
                  csr_matrix(clip_mat[cut:]), txt_te]).tocsr()
    nfeat = dense.shape[1] + clip_mat.shape[1] + len(vec.vocabulary_)
    results.append(evaluate("clip_plus_metadata_cv_text", Xtr, ytr, Xte, yte, nfeat))

    new = pd.DataFrame(results)
    if OUT_CSV.exists():
        combined = pd.concat([pd.read_csv(OUT_CSV), new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nAppended {len(new)} rows -> {OUT_CSV}")

    print("\n" + "=" * 60)
    print("KEY COMPARISON")
    print("  clip_only vs cv_only(0.559): does semantic beat handcrafted?")
    print("  clip_plus_all vs metadata_cv_text(0.580): does CLIP add anything?")
    print("  If all still ~0.58: the ceiling holds even for deep features.")


if __name__ == "__main__":
    main()
