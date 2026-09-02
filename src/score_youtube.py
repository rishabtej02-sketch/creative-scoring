"""Score YouTube training set using cached CLIP → data/youtube_scores.csv"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy import sparse
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from app import cv_features, METADATA_COLS, CV_COLS

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA   = ROOT / "data"
THUMBS = DATA / "thumbnails"
OUT    = DATA / "youtube_scores.csv"

print("load model + tfidf ...")
with open(MODELS/"calibrated_model.pkl","rb") as f: model = pickle.load(f)
with open(MODELS/"tfidf.pkl","rb") as f:            tfidf = pickle.load(f)

print("load cached CLIP ...")
clip_emb = np.load(DATA/"clip_embeddings.npy")
clip_ids = np.load(DATA/"clip_ids.npy", allow_pickle=True)
id2idx = {v:i for i,v in enumerate(clip_ids)}

print("load face detector ...")
onnx = MODELS/"face_detection_yunet_2023mar.onnx"
face = cv2.FaceDetectorYN.create(str(onnx),"",(320,320)) if onnx.exists() else None

print("load features.csv ...")
df = pd.read_csv(DATA/"features.csv", encoding="utf-8")
df = df[df["video_id"].isin(id2idx)].reset_index(drop=True)
print(f"videos to score = {len(df)}")

# titles may live elsewhere — try videos_labeled / videos_raw
title_map = {}
for cand in ["videos_labeled.csv", "videos_raw.csv"]:
    p = DATA/cand
    if p.exists():
        t = pd.read_csv(p, encoding="utf-8", usecols=lambda c: c in ("video_id","title"))
        if "title" in t.columns:
            title_map = dict(zip(t["video_id"], t["title"].fillna("")))
            print(f"titles loaded from {cand} → {len(title_map)}")
            break
if not title_map:
    print("WARN: no title source found → title=''")

out_rows = []
fail = 0
for _, r in tqdm(df.iterrows(), total=len(df), desc="score_yt"):
    vid = r["video_id"]
    tpath = THUMBS/f"{vid}.jpg"
    if not tpath.exists(): fail += 1; continue
    try:
        pil = Image.open(tpath).convert("RGB")
        cv  = cv_features(pil, face)
        meta = {}
        for c in METADATA_COLS:
            if c == "is_short":
                W,H = pil.size; meta[c] = 1.0 if H>W else 0.0
            elif c in r and pd.notna(r[c]):
                meta[c] = float(r[c])
            else:
                meta[c] = 0.0
        dense_vals = [meta[c] for c in METADATA_COLS] + [cv[c] for c in CV_COLS]
        dense = sparse.csr_matrix(np.array(dense_vals, dtype=np.float32).reshape(1,-1))
        clip_v = clip_emb[id2idx[vid]].astype(np.float32).reshape(1,-1)
        clip_s = sparse.csr_matrix(clip_v)
        title = str(title_map.get(vid, ""))
        tf_v = tfidf.transform([title]).astype(np.float32)
        row = sparse.hstack([dense, clip_s, tf_v]).tocsr()
        p = float(model.predict_proba(row)[0,1])
        out_rows.append({"video_id":vid, "score":p,
                         "overperformed": r.get("overperformed"),
                         "labelable": r.get("labelable")})
    except Exception as e:
        fail += 1
        if fail <= 5: print(f"FAIL {vid} → {e}")

pd.DataFrame(out_rows).to_csv(OUT, index=False)
print(f"done → ok={len(out_rows)} fail={fail} → {OUT}")
