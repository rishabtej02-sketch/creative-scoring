"""
gradcam_domain_transfer.py
Sample top-K and bottom-K scored images from each source (YT / Meta / Google),
run Grad-CAM, and assemble one comparison figure showing what the model attends
to across training vs application domains.

Outputs:
  figs/gradcam/{source}_{topbot}_{rank}_{id}.png   (individual side-by-side)
  figs/gradcam_domain_transfer.png                 (grid: 6 rows x K cols, overlay-only)

Usage:
  python src/gradcam_domain_transfer.py           # K=4
  python src/gradcam_domain_transfer.py --k 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# reuse existing code — no drift
from gradcam import build_model_from_ckpt, GradCAM, make_overlay

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIGS = ROOT / "figs" / "gradcam"
FIGS.mkdir(exist_ok=True, parents=True)

CKPT = MODELS / "cnn_gradcam.pt"

# ---- image path resolvers ----
YT_CANDIDATES = [
    ROOT / "data/thumbnails",
    ROOT / "data/youtube/thumbnails",
    ROOT / "data/youtube_thumbs",
    ROOT / "thumbnails",
]

def find_yt_dir():
    for c in YT_CANDIDATES:
        if c.exists() and any(c.glob("*.jpg")):
            return c
    return None

def yt_path(vid, ytdir):
    for ext in (".jpg", ".png", ".jpeg"):
        p = ytdir / f"{vid}{ext}"
        if p.exists():
            return p
    return None

def meta_path(brand, cid):
    for ext in (".jpg", ".png", ".jpeg"):
        p = ROOT / f"data/meta_ads/images/{brand}/{cid}{ext}"
        if p.exists():
            return p
    return None

def google_path(brand, cid):
    for ext in (".jpg", ".png", ".jpeg"):
        p = ROOT / f"data/google_ads/images/{brand}/{cid}{ext}"
        if p.exists():
            return p
    return None


def sample_topbot(df, score_col, k, id_cols):
    """return (top_df, bot_df), each k rows sorted by score desc/asc."""
    d = df.dropna(subset=[score_col]).copy()
    top = d.nlargest(k, score_col)
    bot = d.nsmallest(k, score_col)
    return top, bot


def run_cam_on(img_path, model, cam_engine, tf):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    cam, prob = cam_engine(x)
    combo, overlay = make_overlay(img, cam)
    return img, overlay, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="grid cols per row (headline figure, readability cap)")
    ap.add_argument("--dump-n", type=int, default=25, help="individual overlays per (source x top/bot); total = 6 * dump_n")
    args = ap.parse_args()
    K = args.k
    N = args.dump_n
    assert N >= K, f"--dump-n ({N}) must be >= --k ({K}) so the grid is a subset of the dump"

    # ---- load scores ----
    yt   = pd.read_csv(ROOT / "data/youtube_scores.csv")
    paid = pd.read_csv(ROOT / "data/paid_ad_scores.csv")
    meta_df   = paid[paid["source"] == "meta"].copy()
    google_df = paid[paid["source"] == "google"].copy()
    print(f"scores loaded → yt={len(yt)} meta={len(meta_df)} google={len(google_df)}")
    print(f"dumping N={N} per (source x top/bot) = {6*N} individual overlays; grid uses first K={K} of each")

    # ---- resolve YouTube thumbs dir ----
    ytdir = find_yt_dir()
    if ytdir is None:
        print("ERROR: could not find YouTube thumbnails folder. Tried:")
        for c in YT_CANDIDATES:
            print(f"  - {c}")
        print("Edit YT_CANDIDATES at top of script and re-run.")
        sys.exit(1)
    print(f"YouTube thumbs → {ytdir}")

    # ---- sample per source (N for full dump, K subset for grid) ----
    yt_top, yt_bot     = sample_topbot(yt,        "score", N, ["video_id"])
    m_top,  m_bot      = sample_topbot(meta_df,   "score", N, ["brand", "creative_id"])
    g_top,  g_bot      = sample_topbot(google_df, "score", N, ["brand", "creative_id"])

    # ---- resolve image paths ----
    def resolve(rows, source):
        out = []
        for _, r in rows.iterrows():
            if source == "yt":
                p = yt_path(r["video_id"], ytdir)
                ident = r["video_id"]
            elif source == "meta":
                p = meta_path(r["brand"], str(r["creative_id"]))
                ident = f"{r['brand']}/{r['creative_id']}"
            else:
                p = google_path(r["brand"], str(r["creative_id"]))
                ident = f"{r['brand']}/{r['creative_id']}"
            out.append((p, ident, r["score"]))
        return out

    groups = [
        ("YouTube · TOP",   "yt",     resolve(yt_top, "yt")),
        ("YouTube · BOT",   "yt",     resolve(yt_bot, "yt")),
        ("Meta ads · TOP",  "meta",   resolve(m_top, "meta")),
        ("Meta ads · BOT",  "meta",   resolve(m_bot, "meta")),
        ("Google ads · TOP","google", resolve(g_top, "google")),
        ("Google ads · BOT","google", resolve(g_bot, "google")),
    ]

    # sanity: any missing paths?
    missing = sum(1 for _, _, items in groups for p, _, _ in items if p is None)
    if missing:
        print(f"WARN: {missing} image paths missing — those cells will be blank.")

    # ---- load model + cam ----
    print(f"loading model → {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt)
    mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
    std  = ckpt.get("imagenet_std",  [0.229, 0.224, 0.225])
    size = ckpt.get("input_size", 224)
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    cam_engine = GradCAM(model, model.features[-1])

    # ---- run all, collect overlays ----
    try:
        grid = []  # list of (label, [(overlay_or_None, ident, prob), ...])
        for label, src, items in groups:
            row = []
            for i, (p, ident, score) in enumerate(items):
                if p is None:
                    row.append((None, ident, score))
                    print(f"  [{label}] MISS {ident}")
                    continue
                img, overlay, prob = run_cam_on(p, model, cam_engine, tf)
                # save individual side-by-side too
                combo_path = FIGS / f"{src}_{label.split('·')[1].strip().lower()}_{i+1:02d}_{Path(str(ident)).name}.png"
                combo = Image.new("RGB", (img.width * 2 + 10, img.height), (255, 255, 255))
                combo.paste(img.convert("RGB"), (0, 0))
                combo.paste(overlay, (img.width + 10, 0))
                combo.save(combo_path)
                row.append((overlay, ident, prob))
                print(f"  [{label}] {ident}  csv_score={score:.3f}  cnn_prob={prob:.3f}")
            grid.append((label, row))
    finally:
        cam_engine.close()

    # ---- assemble master figure (first K of each row) ----
    fig, axes = plt.subplots(6, K, figsize=(K * 2.6, 6 * 2.6))
    if K == 1:
        axes = axes.reshape(6, 1)
    for r, (label, row) in enumerate(grid):
        for c, (overlay, ident, prob) in enumerate(row[:K]):
            ax = axes[r, c]
            if overlay is not None:
                ax.imshow(overlay)
                ax.set_title(f"p={prob:.2f}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=60)
    fig.suptitle("Grad-CAM: what the CNN attends to across domains\n"
                 "Row groups: TOP-K vs BOT-K scored images per source", fontsize=12)
    plt.tight_layout()
    out = ROOT / "figs" / "gradcam_domain_transfer.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved grid → {out}")
    print(f"individual overlays → {FIGS}")


if __name__ == "__main__":
    main()
