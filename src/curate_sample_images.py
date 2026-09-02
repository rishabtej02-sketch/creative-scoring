"""
curate_sample_images.py
One-off: for 5 hero brands, pick top-3 + bottom-3 scored ads per (source, brand),
copy the images into data/sample_images/{source}/{brand}/{cid}.{ext}, and write
a slim data/sample_scores.csv for the generalization tab to read on HF Spaces.
"""

import shutil
from pathlib import Path
import pandas as pd

# --- config ---
HERO_BRANDS = ["zomato", "swiggy", "cred", "nykaa", "mamaearth"]
TOP_N = 3
BOT_N = 3

PAID_SCORES  = Path("data/paid_ad_scores.csv")
META_IMG_DIR = Path("data/meta_ads/images")
GOOG_IMG_DIR = Path("data/google_ads/images")

OUT_SCORES = Path("data/sample_scores.csv")
OUT_IMG    = Path("data/sample_images")


def find_source_image(source: str, brand: str, cid: str) -> Path | None:
    root = META_IMG_DIR if source == "meta" else GOOG_IMG_DIR
    for ext in (".jpg", ".png", ".jpeg"):
        p = root / brand / f"{cid}{ext}"
        if p.exists():
            return p
    return None


def main():
    assert PAID_SCORES.exists(), f"missing {PAID_SCORES}"
    df = pd.read_csv(PAID_SCORES)
    print(f"loaded {len(df)} rows from {PAID_SCORES}")

    df = df[df["brand"].isin(HERO_BRANDS)]
    print(f"after brand filter → {len(df)} rows across {df['brand'].nunique()} brands")

    kept_rows   = []
    copied      = 0
    missing     = 0
    OUT_IMG.mkdir(parents=True, exist_ok=True)

    for (source, brand), sub in df.groupby(["source", "brand"]):
        sub = sub.sort_values("score", ascending=False)
        picks = pd.concat([sub.head(TOP_N), sub.tail(BOT_N)]).drop_duplicates("creative_id")
        dest_dir = OUT_IMG / source / brand
        dest_dir.mkdir(parents=True, exist_ok=True)

        for _, r in picks.iterrows():
            src = find_source_image(source, brand, r["creative_id"])
            if src is None:
                missing += 1
                continue
            dst = dest_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
            kept_rows.append(r)

        print(f"  {source}/{brand}: {len(picks)} picks")

    if not kept_rows:
        print("❌ no images found — check paths")
        return

    out_df = pd.DataFrame(kept_rows)
    out_df.to_csv(OUT_SCORES, index=False)

    total_bytes = sum(f.stat().st_size for f in OUT_IMG.rglob("*") if f.is_file())
    print(f"\n✓ copied {copied} images ({missing} missing)")
    print(f"✓ wrote {len(out_df)} rows → {OUT_SCORES}")
    print(f"✓ total sample_images/ size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
