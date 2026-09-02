"""
Step 3c (part 1): Encode every thumbnail into a CLIP embedding, ONCE.

CLIP turns an image into a 512-number vector that captures WHAT is in it
(objects, scene, style) rather than raw pixel statistics. Where the
handcrafted CV features measure brightness/edges/etc., CLIP measures meaning.
Feeding these vectors to the same gradient-boosting model tests whether deep
semantic image content predicts overperformance better than the handcrafted
features did.

This step is the slow one, so it is separate and CACHED. It writes two files:
    data/clip_embeddings.npy   (float32 array, one 512-d row per video)
    data/clip_ids.npy          (video_id order, to line embeddings up later)
On re-run it resumes: already-embedded videos are skipped. Nothing here
touches the label or the split — it is pure feature extraction.

Runs on CPU by default (safe on a small GPU). ~13k thumbnails ≈ 20-40 min
the first time, then instant forever after.

Usage:
  pip install torch transformers pillow numpy pandas
  python embed_clip.py
  python embed_clip.py --limit 50      # quick smoke test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "features.csv"
THUMB_DIR = DATA_DIR / "thumbnails"
EMB_PATH = DATA_DIR / "clip_embeddings.npy"
IDS_PATH = DATA_DIR / "clip_ids.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH = 32


def _to_projected_embeds(out, model):
    """
    Return the PROJECTED image embedding (512-d for ViT-B/32).

    model.get_image_features() already applies the visual projection
    internally — the returned tensor (or wrapper's .image_embeds) IS the
    512-d projected vector. Do NOT run it through model.visual_projection
    again: that double-projects and throws a shape mismatch
    (e.g. 512 in vs a 768-in linear layer expects).
    """
    import torch

    # bare tensor (older transformers): already projected, 512-d
    if torch.is_tensor(out):
        return out.clone()
    # wrapper object (some transformers): .image_embeds is the projected 512-d vector
    if getattr(out, "image_embeds", None) is not None:
        return out.image_embeds.clone()
    # BaseModelOutputWithPooling (this transformers version): get_image_features
    # returns a pooling output whose .pooler_output IS ALREADY the projected
    # 512-d embedding. Verified: pooler_output.shape == [B, 512]. Do NOT run it
    # through visual_projection (that expects a 768-d input -> shape mismatch).
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output.clone()
    raise RuntimeError(f"Unexpected get_image_features output: {type(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N (test)")
    ap.add_argument("--gpu", action="store_true",
                    help="use CUDA if available (risky on <=2GB cards)")
    args = ap.parse_args()

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        sys.exit(f"Missing package: {e}. pip install torch transformers pillow")

    if not IN_CSV.exists():
        sys.exit(f"Not found: {IN_CSV}. Run build_features.py first.")

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}  (CPU is the safe default on a 2GB GPU)")

    df = pd.read_csv(IN_CSV, encoding="utf-8", usecols=["video_id"])
    ids_wanted = df["video_id"].tolist()
    if args.limit:
        ids_wanted = ids_wanted[:args.limit]

    # resume: load whatever is already cached
    done = {}
    if EMB_PATH.exists() and IDS_PATH.exists():
        old_emb = np.load(EMB_PATH)
        old_ids = np.load(IDS_PATH, allow_pickle=True)
        done = {vid: old_emb[i] for i, vid in enumerate(old_ids)}
        print(f"Resuming: {len(done):,} embeddings already cached")

    todo = [v for v in ids_wanted if v not in done]
    print(f"To embed: {len(todo):,} of {len(ids_wanted):,}")
    if not todo:
        print("Nothing to do — all cached.")
        return

    print(f"Loading {MODEL_NAME} ...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    new_emb, new_ids, missing = [], [], 0
    buf_imgs, buf_ids = [], []

    def flush():
        nonlocal missing
        if not buf_imgs:
            return
        inputs = processor(images=buf_imgs, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.get_image_features(**inputs)
        feats = _to_projected_embeds(out, model)
        feats = feats.cpu().numpy().astype("float32")
        new_emb.extend(feats)
        new_ids.extend(buf_ids)
        buf_imgs.clear()
        buf_ids.clear()

    for i, vid in enumerate(todo, 1):
        path = THUMB_DIR / f"{vid}.jpg"
        if not path.exists():
            missing += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            missing += 1
            continue
        buf_imgs.append(img)
        buf_ids.append(vid)

        if len(buf_imgs) >= BATCH:
            flush()
        if i % 500 == 0:
            print(f"  {i}/{len(todo)} | embedded {len(new_ids)} | missing {missing}")

    flush()

    # merge new with cached, write back
    all_ids = list(done.keys()) + new_ids
    all_emb = (list(done.values()) + new_emb) if done else new_emb
    all_emb = np.vstack(all_emb).astype("float32")

    DATA_DIR.mkdir(exist_ok=True)
    np.save(EMB_PATH, all_emb)
    np.save(IDS_PATH, np.array(all_ids, dtype=object))

    print(f"\nWrote {all_emb.shape[0]:,} embeddings x {all_emb.shape[1]} dims")
    print(f"  -> {EMB_PATH}")
    print(f"  -> {IDS_PATH}")
    print(f"Missing/unreadable thumbnails skipped: {missing}")


if __name__ == "__main__":
    main()
