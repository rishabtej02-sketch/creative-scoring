"""
Step 2b: Turn each labeled video into a row of numeric features.

Three feature families, matching the modelling plan:

  METADATA  — available with no image at all. This is the baseline that
              answers "do format and timing predict performance better
              than the creative does?" Nothing here touches a pixel.

  CV BASIC  — cheap handcrafted image statistics: brightness, contrast,
              saturation, colorfulness, edge density, warm/cool balance.
              Fully interpretable ("bright, high-contrast thumbnails win").

  CV FACES/TEXT — face_count + biggest-face area (YuNet, a <1MB CNN that
              ships with OpenCV and is far better than Haar cascades), and
              a text-area estimate via MSER. "Does a face sell the click?"

Reads  data/videos_labeled.csv  and  data/thumbnails/<video_id>.jpg
Writes data/features.csv  (one row per video, label carried through)

Thumbnails that are missing or unreadable still get a metadata row; their
CV columns are left blank so nothing silently becomes zero.

YuNet model file is required for face features. Download once (git-lfs
object, ~230 KB) on your own machine — the URL is printed if it's missing:

  curl -L -o models/face_detection_yunet_2023mar.onnx ^
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Usage:
  pip install opencv-python numpy pandas
  python build_features.py
  python build_features.py --no-faces      # skip face features if model absent
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    sys.exit("Need OpenCV:  pip install opencv-python")

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "videos_labeled.csv"
THUMB_DIR = DATA_DIR / "thumbnails"
OUT_CSV = DATA_DIR / "features.csv"
YUNET_PATH = Path("models") / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Columns we carry straight through from the labeled file (no transform).
PASSTHROUGH = [
    "video_id", "channel_handle", "channel_id", "format",
    "published_at", "view_count", "baseline_median", "prior_count",
    "labelable", "overperformed",
]


# ---------------------------------------------------------------- metadata

def metadata_features(row):
    ts = pd.to_datetime(row["published_at"], utc=True, errors="coerce")
    title = str(row.get("title", "") or "")
    desc = str(row.get("description", "") or "")
    tags = str(row.get("tags", "") or "")

    return {
        "duration_seconds": pd.to_numeric(row.get("duration_seconds"), errors="coerce"),
        "hour_of_day": ts.hour if pd.notna(ts) else np.nan,
        "day_of_week": ts.dayofweek if pd.notna(ts) else np.nan,   # 0=Mon
        "is_weekend": int(ts.dayofweek >= 5) if pd.notna(ts) else np.nan,
        "title_length": len(title),
        "title_word_count": len(title.split()),
        "tag_count": 0 if not tags else len(tags.split("|")),
        "description_length": len(desc),
        "category_id": pd.to_numeric(row.get("category_id"), errors="coerce"),
        "has_caption": 1 if str(row.get("caption")).lower() == "true" else 0,
    }


# ---------------------------------------------------------------- basic CV

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


# ---------------------------------------------------------------- faces

class FaceDetector:
    """YuNet wrapper. Returns (count, biggest_face_area_fraction)."""

    def __init__(self, model_path):
        # input size is reset per image below
        self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320))

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return 0, 0.0
        # faces[:, 0:4] = x, y, w, h
        areas = faces[:, 2] * faces[:, 3]
        biggest = float(areas.max()) / float(w * h)
        return int(len(faces)), biggest


def text_area_fraction(img_bgr):
    """Rough text coverage via MSER regions. Not OCR — just 'busy with text'."""
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


# ---------------------------------------------------------------- driver

def load_image(video_id):
    path = THUMB_DIR / f"{video_id}.jpg"
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img  # None if unreadable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-faces", action="store_true",
                    help="Skip YuNet face features (use if model file absent)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N rows (quick test)")
    args = ap.parse_args()

    if not IN_CSV.exists():
        sys.exit(f"Not found: {IN_CSV}. Run build_labels.py first.")

    df = pd.read_csv(IN_CSV, encoding="utf-8")
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df):,} labeled rows")

    face_det = None
    do_faces = not args.no_faces
    if do_faces:
        if not YUNET_PATH.exists():
            print("\nYuNet model not found at", YUNET_PATH)
            print("Download it once (or re-run with --no-faces):")
            print(f"  mkdir models")
            print(f"  curl -L -o {YUNET_PATH} {YUNET_URL}\n")
            sys.exit("Missing face model.")
        face_det = FaceDetector(YUNET_PATH)

    rows = []
    n_img = n_noimg = 0
    for i, (_, row) in enumerate(df.iterrows(), 1):
        feat = {k: row.get(k) for k in PASSTHROUGH if k in df.columns}
        feat.update(metadata_features(row))

        img = load_image(row["video_id"])
        if img is None:
            feat["has_thumbnail"] = 0
            n_noimg += 1
        else:
            feat["has_thumbnail"] = 1
            n_img += 1
            try:
                feat.update(basic_cv_features(img))
                feat["text_area_frac"] = text_area_fraction(img)
                if do_faces:
                    c, a = face_det.detect(img)
                    feat["face_count"] = c
                    feat["face_area_frac"] = a
            except Exception as e:
                print(f"  CV failed on {row['video_id']}: {e}")

        rows.append(feat)
        if i % 1000 == 0:
            print(f"  {i}/{len(df)} | img {n_img} | no-img {n_noimg}")

    out = pd.DataFrame(rows)
    DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\nWrote {len(out):,} rows, {out.shape[1]} columns -> {OUT_CSV}")
    print(f"Thumbnails used: {n_img:,} | missing/unreadable: {n_noimg:,}")

    feat_cols = [c for c in out.columns if c not in PASSTHROUGH]
    print(f"\nFeature columns ({len(feat_cols)}):")
    print("  " + ", ".join(feat_cols))


if __name__ == "__main__":
    main()
