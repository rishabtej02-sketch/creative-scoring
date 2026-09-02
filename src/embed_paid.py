#!/usr/bin/env python3
"""
embed_paid.py - CLIP embeddings for the Meta + Google ad images.

WHY: data/clip_embeddings.npy covers YouTube only (13,062 x 512). The Meta
     (4,272) and Google (25,224) image folders have no vectors, so retrieval
     is dead on those two tabs. This fills the gap.

VERIFY FIRST. Your stored YouTube vectors were produced by some specific
CLIP call. If this script uses a different one, the new vectors live in a
different space and every cross-platform neighbour lookup is silently wrong.
So --verify re-embeds 200 known thumbnails and compares cosine similarity
against the stored rows. Mean cosine must be > 0.99. Do not skip this.

    python src/embed_paid.py --verify

Then, in a SECOND terminal (this takes 20-40 min on CPU):

    python src/embed_paid.py --run

Resumable. Ctrl-C is safe; re-run continues from the last checkpoint.

OUTPUT:
    data/clip_embeddings_paid.npy   (N, 512) float32
    data/clip_ids_paid.npy          (N,) object, = creative_id
    data/clip_paid_meta.csv         id, platform, path
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time

import numpy as np


def md5_of(path, chunk=1 << 20):
    """Exact file hash. Google Ads Transparency re-lists byte-identical
    creatives under many ad ids; this is how we detect that properly instead
    of inferring it from tied scores."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

EMB_OUT = os.path.join(DATA, "clip_embeddings_paid.npy")
IDS_OUT = os.path.join(DATA, "clip_ids_paid.npy")
META_OUT = os.path.join(DATA, "clip_paid_meta.csv")
CKPT = os.path.join(DATA, "_clip_paid_ckpt.npz")

YT_EMB = os.path.join(DATA, "clip_embeddings.npy")
YT_IDS = os.path.join(DATA, "clip_ids.npy")
THUMBS = os.path.join(DATA, "thumbnails")

SOURCES = [("meta", os.path.join(DATA, "meta_ads")),
           ("google", os.path.join(DATA, "google_ads"))]
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MODEL_NAME = "openai/clip-vit-base-patch32"

_MODEL = _PROC = None


def get_model():
    """Canonical 512-d image embedding: model.get_image_features().

    Note on the pooler_output confusion in the old pipeline:
    vision_model(...).pooler_output is 768-d and still needs visual_projection.
    CLIPVisionModelWithProjection(...).image_embeds is already 512-d.
    get_image_features() does the whole thing correctly in one call, which is
    why it is used here - and why --verify exists to prove it matches.
    """
    global _MODEL, _PROC
    if _MODEL is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        print(f"  loading {MODEL_NAME} (cpu, {torch.get_num_threads()} threads)...")
        _MODEL = CLIPModel.from_pretrained(MODEL_NAME).eval()
        _PROC = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _MODEL, _PROC


def embed_paths(paths, batch=32, show=True, every=10):
    import torch
    from PIL import Image
    model, proc = get_model()
    out, ok_paths, t0 = [], [], time.time()
    nb = (len(paths) + batch - 1) // batch
    for bi in range(nb):
        chunk = paths[bi * batch:(bi + 1) * batch]
        imgs, keep = [], []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                keep.append(p)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt")
            v = model.get_image_features(**inp)
        out.append(v.cpu().numpy().astype(np.float32))
        ok_paths += keep
        if show and (bi % every == 0 or bi == nb - 1):
            done = len(ok_paths)
            el = time.time() - t0
            rate = done / max(el, 1e-6)
            eta = (len(paths) - done) / max(rate, 1e-6)
            print(f"    {done}/{len(paths)}  {rate:.1f} img/s  "
                  f"eta {eta/60:.1f} min", flush=True)
    if not out:
        return np.zeros((0, 512), np.float32), []
    return np.vstack(out), ok_paths


# ---------------------------------------------------------------- verify
def verify(n=200):
    print("=== VERIFY: does get_image_features() match your stored vectors? ===")
    for p in (YT_EMB, YT_IDS):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    E = np.load(YT_EMB).astype(np.float32)
    ids = np.load(YT_IDS, allow_pickle=True)
    print(f"  stored: {E.shape}  ids={len(ids)}")

    if not os.path.isdir(THUMBS):
        sys.exit(f"missing thumbnail dir {THUMBS}")
    disk = {}
    for f in os.listdir(THUMBS):
        if f.lower().endswith(EXTS):
            disk[os.path.splitext(f)[0]] = os.path.join(THUMBS, f)
    print(f"  thumbnails on disk: {len(disk)}")

    idx = [(i, disk[str(v)]) for i, v in enumerate(ids) if str(v) in disk]
    if len(idx) < 20:
        sys.exit("  !! too few thumbnails match clip_ids.npy - check filenames")
    rng = np.random.default_rng(0)
    pick = [idx[i] for i in rng.choice(len(idx), min(n, len(idx)), replace=False)]
    rows = [i for i, _ in pick]
    paths = [p for _, p in pick]
    print(f"  re-embedding {len(paths)} known thumbnails...")

    new, ok = embed_paths(paths, show=True, every=2)
    keep = [paths.index(p) for p in ok]
    old = E[[rows[k] for k in keep]]

    a = new / (np.linalg.norm(new, axis=1, keepdims=True) + 1e-9)
    b = old / (np.linalg.norm(old, axis=1, keepdims=True) + 1e-9)
    cos = (a * b).sum(1)

    print(f"\n  cosine(new, stored): mean={cos.mean():.4f} min={cos.min():.4f} "
          f"p05={np.percentile(cos,5):.4f}")
    print(f"  norms: stored mean={np.linalg.norm(old,axis=1).mean():.2f}  "
          f"new mean={np.linalg.norm(new,axis=1).mean():.2f}")
    if cos.mean() > 0.99:
        print("\n  PASS - same embedding space. Safe to run --run.")
        return True
    print("\n  FAIL - different embedding space.")
    print("  Your existing vectors came from a different CLIP call.")
    print("  Paste the embedding function from your original script and")
    print("  I will match it exactly. Do NOT run --run until this passes.")
    return False


# ---------------------------------------------------------------- run
def collect():
    items = []
    for plat, d in SOURCES:
        if not os.path.isdir(d):
            print(f"  !! missing dir {d}")
            continue
        fs = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
              if p.lower().endswith(EXTS)]
        print(f"  {plat:<8} {len(fs)} images in {os.path.relpath(d, ROOT)}")
        items += [(os.path.splitext(os.path.basename(p))[0], plat, p) for p in fs]
    return items


def run(batch=32, ckpt_every=2000):
    import pandas as pd
    print("=== EMBED PAID ADS ===")
    items = collect()
    if not items:
        sys.exit("no images found")

    done_ids, done_vecs = set(), []
    if os.path.exists(CKPT):
        z = np.load(CKPT, allow_pickle=True)
        done_vecs = [z["vectors"]]
        done_ids = set(z["ids"].tolist())
        print(f"  resuming: {len(done_ids)} already embedded")

    todo = [t for t in items if t[0] not in done_ids]
    print(f"  todo: {len(todo)} of {len(items)}")
    if not todo:
        print("  nothing to do")

    vecs, ids, plats, paths_kept = done_vecs, list(done_ids), [], []
    meta_rows = []
    if os.path.exists(META_OUT):
        meta_rows = pd.read_csv(META_OUT).to_dict("records")

    for i in range(0, len(todo), ckpt_every):
        blk = todo[i:i + ckpt_every]
        print(f"\n  block {i//ckpt_every + 1}/"
              f"{(len(todo)+ckpt_every-1)//ckpt_every}  ({len(blk)} images)")
        V, ok = embed_paths([p for _, _, p in blk], batch=batch)
        stem = {p: (cid, pl) for cid, pl, p in blk}
        for p in ok:
            cid, pl = stem[p]
            ids.append(cid)
            meta_rows.append({"id": cid, "platform": pl,
                              "path": os.path.relpath(p, ROOT),
                              "md5": md5_of(p)})
        vecs.append(V)
        np.savez_compressed(CKPT, vectors=np.vstack(vecs),
                            ids=np.array(ids, dtype=object))
        pd.DataFrame(meta_rows).drop_duplicates("id").to_csv(META_OUT, index=False)
        print(f"    checkpoint saved: {len(ids)} total")

    X = np.vstack(vecs) if vecs else np.zeros((0, 512), np.float32)
    np.save(EMB_OUT, X.astype(np.float32))
    np.save(IDS_OUT, np.array(ids, dtype=object))
    print(f"\n  wrote {os.path.relpath(EMB_OUT, ROOT)}  {X.shape}")
    print(f"  wrote {os.path.relpath(IDS_OUT, ROOT)}  ({len(ids)},)")
    print(f"  wrote {os.path.relpath(META_OUT, ROOT)}")
    print("\n  next: python src/retrieval.py --build")


def hash_only():
    """Just md5 every paid image. ~1-2 min, no torch, no model download.
    Run this FIRST so brand_profiles.py can dedup exactly today, without
    waiting on the full embedding job."""
    import pandas as pd
    print("=== HASH ONLY (fast duplicate detection) ===")
    items = collect()
    rows = []
    t0 = time.time()
    for i, (cid, pl, p) in enumerate(items):
        rows.append({"id": cid, "platform": pl,
                     "path": os.path.relpath(p, ROOT), "md5": md5_of(p)})
        if i % 5000 == 0 and i:
            print(f"    {i}/{len(items)}  {i/(time.time()-t0):.0f} files/s",
                  flush=True)
    df = pd.DataFrame(rows)
    if os.path.exists(META_OUT):
        old = pd.read_csv(META_OUT)
        keep = [c for c in ("id", "platform", "path", "md5") if c in old.columns]
        df = (pd.concat([old[keep], df], ignore_index=True)
                .drop_duplicates("id", keep="last"))
    df.to_csv(META_OUT, index=False)

    print(f"\n  wrote {os.path.relpath(META_OUT, ROOT)}  rows={len(df)}")
    for pl, g in df.groupby("platform"):
        u = g["md5"].nunique()
        print(f"    {pl:<8} files={len(g):<6} unique images={u:<6} "
              f"duplicates={len(g)-u} ({100*(len(g)-u)/max(len(g),1):.1f}%)")
    dup = (df.groupby(["platform", "md5"]).size()
             .sort_values(ascending=False).head(5))
    print("\n  most-repeated single images:")
    for (pl, h), n in dup.items():
        if n > 1:
            print(f"    {pl:<8} {h[:12]}... appears {n}x")
    print("\n  next: python src/brand_profiles.py --audit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-verify", type=int, default=200)
    a = ap.parse_args()
    if a.hash_only:
        hash_only()
    elif a.verify:
        ok = verify(a.n_verify)
        sys.exit(0 if ok else 1)
    elif a.run:
        run(batch=a.batch)
    else:
        ap.print_help()
