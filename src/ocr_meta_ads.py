"""
ocr_meta_ads.py
Fallback for broken Meta ad_creative_body DOM extractor.
Runs easyocr on all Meta ad images → recovers on-image ad text.

Resumable: skips creative_ids already in output CSV.
Appends every 100 rows.

Usage:
  python src/ocr_meta_ads.py            # full run
  python src/ocr_meta_ads.py --limit 20 # smoke test first 20
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import easyocr

ROOT = Path(__file__).resolve().parent.parent
META_IMG = ROOT / "data/meta_ads/images"
OUT = ROOT / "data/meta_ads_ocr.csv"

# ---- args ----
limit = None
if "--limit" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--limit") + 1])

# ---- collect images ----
print(f"scanning {META_IMG} ...")
imgs = sorted(list(META_IMG.rglob("*.jpg")) + list(META_IMG.rglob("*.png")))
print(f"found {len(imgs)} images total")

# ---- resume ----
done = set()
if OUT.exists():
    done = set(pd.read_csv(OUT)["creative_id"].astype(str))
    print(f"resume: {len(done)} already done, skipping")
imgs = [p for p in imgs if p.stem not in done]
print(f"to process: {len(imgs)}")

if limit:
    imgs = imgs[:limit]
    print(f"--limit {limit} → processing {len(imgs)}")

if not imgs:
    print("nothing to do.")
    sys.exit(0)

# ---- reader ----
print("loading easyocr (English) ...")
# Add 'hi' for Hindi/Devanagari if brands use it. English covers Hinglish.
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

# ---- run ----
def flush(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not OUT.exists()
    df.to_csv(OUT, mode="a", header=header, index=False)

rows = []
ok = fail = 0
for i, p in enumerate(tqdm(imgs, desc="ocr")):
    try:
        img = np.array(Image.open(p).convert("RGB"))
        # detail=0 → text only; paragraph=True → merges lines
        result = reader.readtext(img, detail=0, paragraph=True)
        text = " ".join(result).strip()
        ok += 1
    except Exception as e:
        text = ""
        fail += 1
    rows.append({
        "brand": p.parent.name,
        "creative_id": p.stem,
        "ocr_text": text,
        "char_len": len(text),
    })
    if len(rows) >= 100:
        flush(rows)
        rows = []

flush(rows)
print(f"\ndone → ok={ok} fail={fail} → {OUT}")
