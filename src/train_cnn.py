"""Train small CNN for Grad-CAM heatmaps.

NOT the main scorer — the CLIP+LightGBM model (AUC 0.615) stays as the scorer.
This CNN exists only to produce visual explanations (Grad-CAM heatmaps) for the demo.
Expected val AUC: ~0.55-0.60 (weaker than main model — that's fine).

Same time-based split as the main pipeline (sort by published_at, last 20% = test).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
THUMBS = DATA / "thumbnails"
MODELS.mkdir(exist_ok=True)


class ThumbDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = THUMBS / f"{row['video_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(float(row["overperformed"]), dtype=torch.float32)
        return img, label


def load_split(test_frac=0.2, limit=None):
    df = pd.read_csv(DATA / "videos_labeled.csv", encoding="utf-8")
    df = df[df["labelable"] == True].copy()
    df["published_at"] = pd.to_datetime(df["published_at"])
    df = df.sort_values("published_at").reset_index(drop=True)

    # keep only rows whose thumbnail file exists on disk
    before = len(df)
    df = df[df["video_id"].apply(lambda v: (THUMBS / f"{v}.jpg").exists())].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped} rows with missing thumbnails ({before} -> {len(df)})")

    if limit:
        df = df.head(limit).reset_index(drop=True)
        print(f"LIMIT active: using first {limit} rows only (smoke test)")

    n = len(df)
    cut = int(n * (1 - test_frac))
    train_df = df.iloc[:cut].reset_index(drop=True)
    test_df = df.iloc[cut:].reset_index(drop=True)
    return train_df, test_df


def build_model():
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def set_backbone_frozen(model, frozen=True):
    for p in model.features.parameters():
        p.requires_grad = not frozen


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    losses, probs, targets = [], [], []
    desc = "train" if train else "eval "
    for imgs, labels in tqdm(loader, desc=desc):
        imgs = imgs.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(labels.cpu().numpy())
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else float("nan")
    return float(np.mean(losses)), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-epochs", type=int, default=2,
                    help="epochs to keep backbone frozen at start")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only first N rows (smoke test)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="quick sanity run: 500 rows, 1 epoch, batch 8")
    args = ap.parse_args()

    if args.smoke_test:
        args.limit = 500
        args.epochs = 1
        args.batch = 8
        print("SMOKE TEST mode\n")

    device = torch.device("cpu")
    print(f"device: {device}")

    train_df, test_df = load_split(limit=args.limit)
    print(f"train: {len(train_df)}  test: {len(test_df)}")
    print(f"train win rate: {train_df['overperformed'].mean():.3f}")
    print(f"test win rate:  {test_df['overperformed'].mean():.3f}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_ds = ThumbDataset(train_df, train_tf)
    test_ds = ThumbDataset(test_df, eval_tf)
    # num_workers=0 on Windows (multiprocessing safer)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    history = []
    optimizer = None

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_epochs:
            set_backbone_frozen(model, frozen=True)
            if optimizer is None or epoch == 1:
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=args.lr
                )
            print(f"epoch {epoch}: backbone FROZEN (head only, lr={args.lr})")
        else:
            set_backbone_frozen(model, frozen=False)
            if epoch == args.freeze_epochs + 1:
                # switching to full fine-tune, lower lr to avoid destroying pretrained weights
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
                print(f"epoch {epoch}: backbone UNFROZEN (fine-tune, lr={args.lr*0.1})")
            else:
                print(f"epoch {epoch}: fine-tune continues (lr={args.lr*0.1})")

        train_loss, train_auc = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        val_loss, val_auc = run_epoch(model, test_dl, criterion, optimizer, device, train=False)
        print(f"epoch {epoch}: train loss={train_loss:.4f} auc={train_auc:.4f}  "
              f"val loss={val_loss:.4f} auc={val_auc:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_auc": train_auc,
            "val_loss": val_loss, "val_auc": val_auc,
        })

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "state_dict": model.state_dict(),
                "arch": "mobilenet_v3_small",
                "val_auc": val_auc,
                "epoch": epoch,
                "input_size": 224,
                "imagenet_mean": imagenet_mean,
                "imagenet_std": imagenet_std,
            }, MODELS / "cnn_gradcam.pt")
            print(f"  -> saved best (val auc {val_auc:.4f})")

    with open(MODELS / "cnn_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nDONE. best val auc: {best_auc:.4f}")
    print(f"checkpoint: {MODELS / 'cnn_gradcam.pt'}")


if __name__ == "__main__":
    main()
