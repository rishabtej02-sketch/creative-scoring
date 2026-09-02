"""Download full-res ad images, skip s60x60 thumbnails."""
import json, glob, os, time, re, argparse, urllib.request, urllib.error
from pathlib import Path

OUT_DIR = Path("data/meta_ads/images")
JSON_DIR = Path("data/meta_ads")
DELAY = 0.3

def is_thumbnail(url: str) -> bool:
    """Skip tiny 60x60 and profile pic images."""
    if 's60x60' in url or 's148x148' in url or 's206x206' in url:
        return True
    if 'profile' in url.lower():
        return True
    return False

def download_one(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 5000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if len(data) < 2000:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    json_files = sorted(JSON_DIR.glob("*.json"))
    if args.smoke_test:
        json_files = json_files[:3]

    total_ok, total_fail, total_skip, total_thumb = 0, 0, 0, 0

    for jf in json_files:
        data = json.load(open(jf))
        brand = data.get("brand_slug") or jf.stem
        ads = data.get("ads", [])
        brand_dir = OUT_DIR / brand
        ok, fail, skip, thumb = 0, 0, 0, 0

        for ad in ads:
            ad_id = ad.get("ad_archive_id", "unknown")
            images = ad.get("creative_images", [])
            img_idx = 0
            for img_url in images:
                if is_thumbnail(img_url):
                    thumb += 1
                    continue
                dest = brand_dir / f"{ad_id}_{img_idx}.jpg"
                img_idx += 1
                if dest.exists() and dest.stat().st_size > 5000:
                    skip += 1
                    continue
                if download_one(img_url, dest):
                    ok += 1
                else:
                    fail += 1
                if ok % 10 == 0 and ok > 0:
                    print(f"  {brand}: {ok} downloaded...", flush=True)
                time.sleep(DELAY)

        total_ok += ok; total_fail += fail; total_skip += skip; total_thumb += thumb
        print(f"{brand:>20}: {ok:>4} new | {skip:>4} skip | {fail:>4} fail | {thumb:>4} thumbs_skipped")

    print(f"\n{'TOTAL':>20}: {total_ok:>4} new | {total_skip:>4} skip | {total_fail:>4} fail | {total_thumb:>4} thumbs_skipped")

if __name__ == "__main__":
    main()
