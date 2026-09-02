"""Grad-CAM heatmap generator for the trained CNN.

Loads models/cnn_gradcam.pt, produces a heatmap overlay showing where the CNN
"looked" when scoring a thumbnail.

IMPORTANT: shows where the model looked, NOT what to change. Still correlation,
not instruction. Frame it that way in the report / demo.

Usage:
    # single image
    python src/gradcam.py path/to/thumb.jpg
    python src/gradcam.py path/to/thumb.jpg --out out.png

    # batch (folder of images)
    python src/gradcam.py --batch data/thumbnails --out-dir data/heatmaps --limit 20
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT_CKPT = MODELS / "cnn_gradcam.pt"


def build_model_from_ckpt(ckpt):
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


class GradCAM:
    """Grad-CAM on a target conv layer.
    For MobileNetV3-Small, target = model.features[-1] (final conv block, 7x7 feature map).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._save_activations)
        self.h2 = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, input_tensor):
        """input_tensor: (1, 3, H, W). Returns (cam_2d_np in [0,1], prob)."""
        self.model.zero_grad()
        logit = self.model(input_tensor)         # (1, 1)
        score = logit.squeeze()
        score.backward()

        # activations: (1, C, h, w). gradients: (1, C, h, w).
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)     # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        # normalize to [0, 1]
        cam = cam - cam.min()
        maxv = cam.max()
        if maxv > 0:
            cam = cam / maxv
        return cam.squeeze().cpu().numpy(), float(torch.sigmoid(score).item())


def make_overlay(orig_img: Image.Image, cam: np.ndarray, alpha=0.45):
    """orig_img: PIL RGB. cam: 2D array in [0,1]. Returns (side_by_side, overlay_only)."""
    W, H = orig_img.size
    # upsample cam to image size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0
    # colormap: jet (classic heatmap, red = high attention)
    heat = (cm.jet(cam_arr)[..., :3] * 255).astype(np.uint8)  # drop alpha channel
    heat_img = Image.fromarray(heat)
    overlay = Image.blend(orig_img.convert("RGB"), heat_img, alpha=alpha)
    # side-by-side: original | overlay (10px white gutter)
    combo = Image.new("RGB", (W * 2 + 10, H), (255, 255, 255))
    combo.paste(orig_img, (0, 0))
    combo.paste(overlay, (W + 10, 0))
    return combo, overlay


def process(image_path, model, cam_engine, transform):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    cam, prob = cam_engine(x)
    combo, overlay = make_overlay(img, cam)
    return combo, overlay, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="path to a thumbnail (single-image mode)")
    ap.add_argument("--out", default=None, help="output path (single-image mode)")
    ap.add_argument("--batch", default=None, help="folder of images (batch mode)")
    ap.add_argument("--out-dir", default=None, help="output folder (batch mode)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of batch images")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    val_auc = ckpt.get("val_auc", float("nan"))
    print(f"loaded checkpoint: {args.ckpt}  (val_auc={val_auc:.4f})")

    model = build_model_from_ckpt(ckpt)
    mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("imagenet_std", [0.229, 0.224, 0.225])
    size = ckpt.get("input_size", 224)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # target = last conv block for MobileNetV3-Small
    target_layer = model.features[-1]
    cam_engine = GradCAM(model, target_layer)

    try:
        if args.batch:
            in_dir = Path(args.batch)
            out_dir = Path(args.out_dir) if args.out_dir else in_dir.parent / "heatmaps"
            out_dir.mkdir(exist_ok=True, parents=True)
            files = sorted(list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")))
            if args.limit:
                files = files[:args.limit]
            print(f"batch: {len(files)} images -> {out_dir}")
            for i, f in enumerate(files, 1):
                combo, overlay, prob = process(f, model, cam_engine, tf)
                combo.save(out_dir / f"{f.stem}_gradcam.png")
                print(f"  [{i}/{len(files)}] {f.name}  prob={prob:.3f}")
            print(f"\nDONE. wrote {len(files)} overlays to {out_dir}")
        else:
            if not args.image:
                ap.error("provide an image path or use --batch <folder>")
            img_path = Path(args.image)
            out = Path(args.out) if args.out else img_path.parent / f"{img_path.stem}_gradcam.png"
            combo, overlay, prob = process(img_path, model, cam_engine, tf)
            combo.save(out)
            print(f"prob = {prob:.3f}  (probability this beats channel baseline)")
            print(f"saved: {out}")
    finally:
        cam_engine.close()


if __name__ == "__main__":
    main()
