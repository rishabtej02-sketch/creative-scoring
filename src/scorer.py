#!/usr/bin/env python3
"""
scorer.py - the ONE inference path. Nothing else may extract CV features.

WHY THIS FILE EXISTS:
app.py grew its own copy of the CV feature extractor. It drifted from
build_features.py, which is what actually produced the training matrix.
Six of nine CV features diverged:

    feature          training (build_features.py)   app.py (drifted)
    brightness       gray.mean()        0-255       /255  -> 0-1
    contrast         gray.std()         0-255       /255  -> 0-1
    saturation       hsv[:,:,1].mean()  0-255       /255  -> 0-1
    warm_ratio       r.mean()/b.mean()  ratio       warm-hue pixel fraction
    text_area_frac   MSER union mask, AR + size     raw sum of bboxes,
                     filtered, then mask.mean()     overlaps double-counted
    face_area_frac   BIGGEST face / area            SUM of all faces / area

Training and evaluation were never affected - AUC 0.615 was measured on
features.csv, which is correct. Only inference was wrong. Three features
were constant-shifted (every tree split lands the same side, so they carry
no signal), one measured a different quantity, and two used a different
algorithm. Scores still looked plausible because CLIP and TF-IDF carried
the prediction.

The formulas below are copied verbatim from build_features.py. If the two
ever disagree, build_features.py wins - it defines the training data.
verify_parity.py proves they agree, the same way embed_paid.py --verify
proved the CLIP embeddings matched (cosine 1.0000) before the long run.

Usage:
    python src/scorer.py --score data/sample_images/foo.jpg --title "hello"
    python src/scorer.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

YUNET_PATH = MODELS / "face_detection_yunet_2023mar.onnx"

# --- feature spec (must match training order exactly) ---
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


# ================================================================ CV FEATURES
# Everything in this block is verbatim from build_features.py. Do not
# "improve" it. Any change here silently invalidates every stored score.

def colorfulness(img_bgr):
    """Hasler & Susstrunk (2003) colorfulness metric."""
    b, g, r = cv2.split(img_bgr.astype("float32"))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return std + 0.3 * mean


def basic_cv_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(img_bgr.astype("float32"))
    edges = cv2.Canny(gray, 100, 200)

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[:, :, 1].mean()),
        "colorfulness": float(colorfulness(img_bgr)),
        "edge_density": float((edges > 0).mean()),
        "warm_ratio": float(r.mean() / (b.mean() + 1e-6)),
    }


def text_area_fraction(img_bgr):
    """Rough text coverage via MSER regions. Not OCR - just 'busy with text'.

    NOTE this returns a FRACTION 0-1. platform_specs.py expects PERCENT.
    Multiply by 100 at that boundary, never here."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    if not regions:
        return 0.0
    mask = np.zeros(gray.shape, dtype="uint8")
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        ar = w / (h + 1e-6)
        # text-ish boxes: wider-than-tall-ish, not huge blobs
        if 0.2 < ar < 10 and (w * h) < 0.25 * gray.size:
            mask[y:y + h, x:x + w] = 1
    return float(mask.mean())


class FaceDetector:
    """YuNet wrapper. Returns (count, biggest_face_area_fraction).

    BIGGEST, not sum. app.py summed all faces - a group shot inflated the
    fraction well past anything in the training distribution."""

    def __init__(self, model_path=YUNET_PATH):
        self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320))

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return 0, 0.0
        areas = faces[:, 2] * faces[:, 3]
        biggest = float(areas.max()) / float(w * h)
        return int(len(faces)), biggest


def cv_features(img_bgr, face_det=None) -> dict:
    """All nine CV_COLS for one BGR image. Order-independent dict."""
    feat = basic_cv_features(img_bgr)
    feat["text_area_frac"] = text_area_fraction(img_bgr)
    if face_det is not None:
        c, a = face_det.detect(img_bgr)
        feat["face_count"] = float(c)
        feat["face_area_frac"] = float(a)
    else:
        feat["face_count"] = 0.0
        feat["face_area_frac"] = 0.0
    return feat


# ================================================================ IMAGE IO
def load_bgr(src) -> np.ndarray:
    """Accepts a path, raw bytes, a file-like object, or a PIL image.

    Decoding goes through cv2 wherever possible so the pixel array matches
    what build_features.py saw via cv2.imread. PIL and OpenCV can disagree
    by a level or two on the same JPEG."""
    if isinstance(src, (str, os.PathLike, Path)):
        img = cv2.imread(str(src))
        if img is None:
            raise ValueError(f"unreadable image: {src}")
        return img
    if isinstance(src, (bytes, bytearray)):
        buf = np.frombuffer(bytes(src), dtype=np.uint8)
    elif hasattr(src, "read"):
        try:
            src.seek(0)
        except Exception:
            pass
        buf = np.frombuffer(src.read(), dtype=np.uint8)
    else:  # PIL fallback
        arr = np.array(src.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def bgr_to_pil(img_bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ================================================================ MODELS
class Bundle:
    """Everything needed to score. Built once, passed around."""

    def __init__(self, model, tfidf, spec, clip_model, clip_proc,
                 face_det, defaults):
        self.model = model
        self.tfidf = tfidf
        self.spec = spec
        self.clip_model = clip_model
        self.clip_proc = clip_proc
        self.face_det = face_det
        self.defaults = defaults


def load_defaults() -> dict:
    """Median/mode metadata values from the training set, for the fields a
    user cannot supply at upload time (duration, category, posting hour)."""
    df = pd.read_csv(DATA / "features.csv", encoding="utf-8")
    d = {}
    for col in METADATA_COLS:
        if col not in df.columns:
            d[col] = 0.0
            continue
        if col in ("is_weekend", "has_caption", "is_short"):
            try:
                d[col] = float(df[col].mode().iloc[0])
            except Exception:
                d[col] = 0.0
        else:
            d[col] = float(df[col].median())
    return d


def load_all(with_clip=True, verbose=False) -> Bundle:
    with open(MODELS / "calibrated_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(MODELS / "feature_spec.json", encoding="utf-8") as f:
        spec = json.load(f)

    clip_model = clip_proc = None
    if with_clip:
        if verbose:
            print("  loading openai/clip-vit-base-patch32 (cpu)...")
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

    face_det = FaceDetector() if YUNET_PATH.exists() else None
    if face_det is None and verbose:
        print(f"  [warn] YuNet missing at {YUNET_PATH} - face features = 0")

    return Bundle(model, tfidf, spec, clip_model, clip_proc,
                  face_det, load_defaults())


def clip_embedding(pil_img, clip_model, clip_proc) -> np.ndarray:
    """get_image_features() is the verified call - embed_paid.py --verify
    reproduced stored vectors at cosine 1.0000 using exactly this."""
    import torch
    with torch.no_grad():
        inputs = clip_proc(images=pil_img.convert("RGB"), return_tensors="pt")
        emb = clip_model.get_image_features(**inputs)
    return emb.squeeze().cpu().numpy()


# ================================================================ SCORING
def build_feature_row(title, img_bgr, bundle: Bundle, overrides=None):
    """hstack([dense(METADATA_COLS + CV_COLS), clip_512, tfidf(title)])
    The order is strict. It is the training order. Do not reorder."""
    overrides = overrides or {}
    meta = {}
    H, W = img_bgr.shape[:2]
    for col in METADATA_COLS:
        if col in overrides:
            meta[col] = float(overrides[col])
        elif col == "title_length":
            meta[col] = float(len(title))
        elif col == "title_word_count":
            meta[col] = float(len(title.split()))
        elif col == "is_short":
            meta[col] = 1.0 if H > W else 0.0
        else:
            meta[col] = float(bundle.defaults.get(col, 0.0))

    cv = cv_features(img_bgr, bundle.face_det)

    dense_vals = [meta[c] for c in METADATA_COLS] + [cv[c] for c in CV_COLS]
    dense = sparse.csr_matrix(
        np.array(dense_vals, dtype=np.float32).reshape(1, -1))
    pil = bgr_to_pil(img_bgr)
    clip_vec = clip_embedding(pil, bundle.clip_model,
                              bundle.clip_proc).astype(np.float32).reshape(1, -1)
    tfidf_vec = bundle.tfidf.transform([title]).astype(np.float32)
    row = sparse.hstack([dense, sparse.csr_matrix(clip_vec),
                         tfidf_vec]).tocsr()
    return row, meta, cv, clip_vec.ravel()


def score_image(src, title="", bundle: Bundle = None, overrides=None):
    """Returns (score, meta, cv, clip_vec). `title` is "" for paid creatives -
    ad_text is legal disclosure, not YouTube-equivalent title signal."""
    bundle = bundle or load_all()
    img = load_bgr(src)
    row, meta, cv, clip_vec = build_feature_row(title, img, bundle, overrides)
    score = float(bundle.model.predict_proba(row)[:, 1][0])
    return score, meta, cv, clip_vec


# ================================================================ CLI
def _self_test():
    """Extract CV features from a handful of training thumbnails and compare
    against the stored features.csv values. Same idea as the CLIP --verify."""
    fp = DATA / "features.csv"
    if not fp.exists():
        print(f"missing {fp}")
        return 1
    df = pd.read_csv(fp, encoding="utf-8")
    if "has_thumbnail" in df.columns:
        df = df[df["has_thumbnail"] == 1]
    df = df.dropna(subset=["brightness"]).head(10)
    face_det = FaceDetector() if YUNET_PATH.exists() else None

    worst = {c: 0.0 for c in CV_COLS}
    n = 0
    for _, r in df.iterrows():
        p = DATA / "thumbnails" / f"{r['video_id']}.jpg"
        if not p.exists():
            continue
        got = cv_features(load_bgr(p), face_det)
        n += 1
        for c in CV_COLS:
            if c in r and pd.notna(r[c]):
                worst[c] = max(worst[c], abs(float(got[c]) - float(r[c])))

    print(f"self-test on {n} thumbnails - max abs diff vs features.csv:")
    bad = False
    for c in CV_COLS:
        flag = "OK  " if worst[c] < 1e-6 else "DIFF"
        if worst[c] >= 1e-6:
            bad = True
        print(f"  [{flag}] {c:<16} {worst[c]:.3e}")
    print("PARITY FAIL" if bad else "PARITY OK - extractor matches training")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", metavar="IMG", help="score one image")
    ap.add_argument("--title", default="", help="title text for the scorer")
    ap.add_argument("--self-test", action="store_true",
                    help="check CV parity against features.csv")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(_self_test())
    if a.score:
        b = load_all(verbose=True)
        s, meta, cv, _ = score_image(a.score, a.title, b)
        print(f"score = {s:.4f}")
        print("CV  :", {k: round(v, 4) for k, v in cv.items()})
        print("META:", {k: round(v, 2) for k, v in meta.items()})
    else:
        ap.print_help()
