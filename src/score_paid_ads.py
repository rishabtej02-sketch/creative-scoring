"""Score paid ads with calibrated model. Meta + Google → data/paid_ad_scores.csv"""
import sys, csv, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.sparse import vstack
from tqdm import tqdm
import cv2
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
from app import build_feature_row, METADATA_COLS

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA   = ROOT / "data"
OUT    = DATA / "paid_ad_scores.csv"
BATCH  = 32

SOURCES = {
    "meta":   DATA / "meta_ads"   / "images",
    "google": DATA / "google_ads" / "images",
}

def load_all():
    print("load model + tfidf + spec ...")
    with open(MODELS / "calibrated_model.pkl", "rb") as f: model = pickle.load(f)
    with open(MODELS / "tfidf.pkl", "rb") as f:            tfidf = pickle.load(f)
    with open(MODELS / "feature_spec.json") as f:          spec  = json.load(f)

    print("load CLIP ...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"); clip_model.eval()
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("load face detector ...")
    onnx = MODELS / "face_detection_yunet_2023mar.onnx"
    face = cv2.FaceDetectorYN.create(str(onnx), "", (320, 320)) if onnx.exists() else None

    print("load defaults ...")
    df = pd.read_csv(DATA / "features.csv", encoding="utf-8")
    defaults = {}
    for col in METADATA_COLS:
        if col not in df.columns: defaults[col] = 0.0; continue
        if col in ("is_weekend","has_caption","is_short"):
            defaults[col] = float(df[col].mode().iloc[0])
        else:
            defaults[col] = float(df[col].median())
    return model, tfidf, clip_model, clip_proc, face, defaults

def collect():
    for src, root in SOURCES.items():
        if not root.exists():
            print(f"skip {src} → {root} missing"); continue
        for p in root.rglob("*.jpg"):
            yield src, p.parent.name, p.stem, p

def main():
    model, tfidf, clip_model, clip_proc, face, defaults = load_all()
    jobs = list(collect())
    print(f"total imgs = {len(jobs)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    f = OUT.open("w", newline="", encoding="utf-8")
    w = csv.writer(f); w.writerow(["source","brand","creative_id","score"])

    buf_meta, buf_rows = [], []
    ok = fail = 0

    def flush():
        nonlocal ok
        if not buf_rows: return
        X = vstack(buf_rows).tocsr()
        p = model.predict_proba(X)[:, 1]
        for (s,b,c), score in zip(buf_meta, p):
            w.writerow([s, b, c, f"{score:.6f}"])
        ok += len(buf_rows)
        buf_meta.clear(); buf_rows.clear()

    for src, brand, cid, path in tqdm(jobs, desc="score"):
        try:
            pil = Image.open(path).convert("RGB")
            row, _, _ = build_feature_row("", pil, defaults, tfidf,
                                          clip_model, clip_proc, face)
            buf_meta.append((src, brand, cid))
            buf_rows.append(row)
            if len(buf_rows) >= BATCH: flush()
        except Exception as e:
            fail += 1
            if fail <= 5: print(f"FAIL {path.name} → {e}")

    flush(); f.close()
    print(f"done → ok={ok} fail={fail} → {OUT}")

if __name__ == "__main__":
    main()
