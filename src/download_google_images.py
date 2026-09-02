"""Download Google Ads Transparency Center thumbnails (image ads only)."""
import json, time, argparse, urllib.request, urllib.error
from pathlib import Path

OUT_DIR = Path("data/google_ads/images")
JSON_DIR = Path("data/google_ads")
DELAY = 0.3

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
    total_ok, total_fail, total_skip, total_video = 0, 0, 0, 0
    for jf in json_files:
        data = json.load(open(jf))
        brand = data.get("slug") or jf.stem
        ads = data.get("ads", [])
        brand_dir = OUT_DIR / brand
        ok, fail, skip, video = 0, 0, 0, 0
        for ad in ads:
            if ad.get("format") != "image":
                video += 1
                continue
            url = ad.get("thumbnail_url", "")
            if not url:
                fail += 1
                continue
            cid = ad.get("creative_id", "unknown")
            dest = brand_dir / f"{cid}.jpg"
            if dest.exists() and dest.stat().st_size > 5000:
                skip += 1
                continue
            if download_one(url, dest):
                ok += 1
            else:
                fail += 1
            if ok % 10 == 0 and ok > 0:
                print(f"  {brand}: {ok} downloaded...", flush=True)
            time.sleep(DELAY)
        total_ok += ok; total_fail += fail; total_skip += skip; total_video += video
        print(f"{brand:>25}: {ok:>4} new | {skip:>4} skip | {fail:>4} fail | {video:>4} video_skipped")
    print(f"\n{'TOTAL':>25}: {total_ok:>4} new | {total_skip:>4} skip | {total_fail:>4} fail | {total_video:>4} video_skipped")

if __name__ == "__main__":
    main()