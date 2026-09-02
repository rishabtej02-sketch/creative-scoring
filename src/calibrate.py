"""
Step 4: Calibrate the best model (clip + metadata + cv + text) into HONEST
probabilities -- WITHOUT sacrificing training data.

The model outputs a score. In the allocation step that score becomes money,
so it must mean what it says: when it says 0.70, ~70% of those creatives
should actually overperform. Raw gradient-boosting scores usually are not
calibrated this way.

Naive fix (hold out a calibration slice) costs training rows and drops AUC.
Instead we use CalibratedClassifierCV with cv=5: it does internal k-fold
cross-fitting on the SAME train set -- each fold's calibrator is fit on rows
the model did not see in that fold, then averaged. No data is thrown away.
The final model still trains on the full train split, so headline AUC is
preserved AND probabilities come out calibrated.

Split (same time order as train_clip.py, leak-safe):
    train  (oldest 80%)  -> fit model + calibrate via internal 5-fold CV
    test   (newest 20%)  -> measure AUC + calibration on unseen data

Metrics on test:
    AUC   -> ranking quality (should match train_clip.py's ~0.615)
    Brier -> squared error of probabilities (lower = better)
    ECE   -> expected calibration error across bins (lower = trustworthy)

Outputs (for the Streamlit app):
    models/calibrated_model.pkl  (fitted CalibratedClassifierCV -- predict_proba is calibrated)
    models/tfidf.pkl             (fitted title vectorizer)
    models/feature_spec.json     (column order, so the app rebuilds X identically)
    data/calibration_report.csv  (before/after metrics)
    data/reliability_calibrated.csv

Usage:
  python calibrate.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
IN_CSV = DATA_DIR / "features.csv"
RAW_CSV = DATA_DIR / "videos_raw.csv"
EMB_PATH = DATA_DIR / "clip_embeddings.npy"
IDS_PATH = DATA_DIR / "clip_ids.npy"

RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 300
TEST_FRACTION = 0.20
CALIB_METHOD = "isotonic"   # "isotonic" (flexible) or "sigmoid" (Platt, smoother on small data)
CALIB_CV = 5

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
            sys.exit(f"Not found: {p}. Run build_features.py and embed_clip.py first.")

    df = pd.read_csv(IN_CSV, encoding="utf-8")
    titles = pd.read_csv(RAW_CSV, encoding="utf-8", usecols=["video_id", "title"])
    df = df.merge(titles, on="video_id", how="left")
    df["title"] = df["title"].fillna("").astype(str)

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


def _brier(y, p):
    return float(np.mean((p - y) ** 2))


def _ece(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _reliability(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        rows.append({
            "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
            "n": int(m.sum()),
            "mean_pred": round(float(p[m].mean()), 4) if m.sum() else None,
            "frac_pos": round(float(y[m].mean()), 4) if m.sum() else None,
        })
    return pd.DataFrame(rows)


def main():
    df, clip_mat = load()
    y = df["overperformed"].values
    n = len(df)
    cut = int(n * (1 - TEST_FRACTION))
    ytr, yte = y[:cut], y[cut:]

    print(f"  train {cut:,} (older) | test {n - cut:,} (newer)")
    print(f"  split at {df['published_at'].iloc[cut].date()}\n")

    dense = df[METADATA_COLS + CV_COLS].apply(pd.to_numeric, errors="coerce").fillna(-1)
    vec = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, stop_words="english",
                          ngram_range=(1, 2), min_df=5)
    txt_tr = vec.fit_transform(df["title"].iloc[:cut])
    txt_te = vec.transform(df["title"].iloc[cut:])

    Xtr = hstack([csr_matrix(dense.iloc[:cut].values),
                  csr_matrix(clip_mat[:cut]), txt_tr]).tocsr()
    Xte = hstack([csr_matrix(dense.iloc[cut:].values),
                  csr_matrix(clip_mat[cut:]), txt_te]).tocsr()

    base = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1,
    )

    # ---- RAW baseline: plain model, for before/after comparison -----------
    base.fit(Xtr, ytr)
    p_te_raw = base.predict_proba(Xte)[:, 1]

    # ---- CALIBRATED: 5-fold cross-fit calibration on the SAME train set ----
    # each fold trains on 4/5, calibrates on the held-out 1/5, then averages.
    # no rows are sacrificed; the ensemble trains on all of train.
    cal = CalibratedClassifierCV(base, method=CALIB_METHOD, cv=CALIB_CV)
    cal.fit(Xtr, ytr)
    p_te_cal = cal.predict_proba(Xte)[:, 1]

    rep = pd.DataFrame([
        {"metric": "AUC",
         "raw": round(roc_auc_score(yte, p_te_raw), 4),
         "calibrated": round(roc_auc_score(yte, p_te_cal), 4)},
        {"metric": "Brier",
         "raw": round(_brier(yte, p_te_raw), 4),
         "calibrated": round(_brier(yte, p_te_cal), 4)},
        {"metric": "ECE",
         "raw": round(_ece(yte, p_te_raw), 4),
         "calibrated": round(_ece(yte, p_te_cal), 4)},
        {"metric": "acc@0.5",
         "raw": round(accuracy_score(yte, (p_te_raw >= 0.5).astype(int)), 4),
         "calibrated": round(accuracy_score(yte, (p_te_cal >= 0.5).astype(int)), 4)},
    ])
    print("=" * 56)
    print(f"CALIBRATION ({CALIB_METHOD}, {CALIB_CV}-fold) -- test set (newest, unseen)")
    print(rep.to_string(index=False))
    print("  AUC: should match train_clip.py (~0.615) -- no train rows lost.")
    print("  Brier & ECE: lower after = probabilities now trustworthy.")

    reli = _reliability(yte, p_te_cal)
    print("\nReliability (calibrated) -- mean_pred should track frac_pos:")
    print(reli.to_string(index=False))

    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "calibrated_model.pkl", "wb") as f:
        pickle.dump(cal, f)
    with open(MODELS_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(MODELS_DIR / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump({"metadata_cols": METADATA_COLS, "cv_cols": CV_COLS,
                   "clip_dims": int(clip_mat.shape[1]),
                   "tfidf_max_features": MAX_TFIDF_FEATURES,
                   "calib_method": CALIB_METHOD, "calib_cv": CALIB_CV,
                   "order": "dense(metadata+cv) | clip | tfidf(title)"}, f, indent=2)
    rep.to_csv(DATA_DIR / "calibration_report.csv", index=False, encoding="utf-8")
    reli.to_csv(DATA_DIR / "reliability_calibrated.csv", index=False, encoding="utf-8")

    print("\nSaved:")
    for p in ["calibrated_model.pkl", "tfidf.pkl", "feature_spec.json"]:
        print(f"  -> {MODELS_DIR / p}")
    print(f"  -> {DATA_DIR / 'calibration_report.csv'}")
    print("\nApp note: load calibrated_model.pkl -> predict_proba is ALREADY calibrated.")


if __name__ == "__main__":
    main()
