"""
Step 1: Collect branded video metadata from the YouTube Data API v3.

Design notes (these matter for the report):
  - We never call search.list (100 quota units). We resolve each channel
    once via channels.list (1 unit) to get its "uploads" playlist, then page
    through that playlist with playlistItems.list (1 unit per 50 videos).
    Statistics come from videos.list batched 50 IDs at a time (1 unit).
    Cost per channel with ~300 videos: about 13 units out of 10,000/day.

  - Everything is cached to disk. Raw API responses are written per channel
    under data/raw/, so a re-run skips channels already collected. If you hit
    the daily quota the script stops cleanly and you resume tomorrow.

  - Thumbnails come from i.ytimg.com, which costs ZERO API quota. That step
    is separate so you can run it independently of the metadata pull.

Usage:
    export YOUTUBE_API_KEY="your_key_here"        # Windows: set YOUTUBE_API_KEY=...
    pip install requests

    python collect_youtube.py --smoke-test        # 3 channels, verify setup
    python collect_youtube.py                     # full metadata pull
    python collect_youtube.py --thumbnails        # download thumbnail images
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
THUMB_DIR = DATA_DIR / "thumbnails"
VIDEOS_CSV = DATA_DIR / "videos_raw.csv"
CHANNELS_CSV = DATA_DIR / "channels_resolved.csv"

MAX_VIDEOS_PER_CHANNEL = 300   # newest N uploads per brand
DAILY_QUOTA = 10000
QUOTA_SAFETY_MARGIN = 200      # stop before hitting the hard wall


class QuotaExhausted(Exception):
    """Raised when the API reports the daily quota is spent."""


class Quota:
    """Tracks estimated quota spend so we can stop before Google does."""

    def __init__(self, budget=DAILY_QUOTA - QUOTA_SAFETY_MARGIN):
        self.spent = 0
        self.budget = budget

    def charge(self, units):
        if self.spent + units > self.budget:
            raise QuotaExhausted(
                f"Local quota budget reached ({self.spent}/{self.budget} units)."
            )
        self.spent += units

    def report(self):
        pct = 100 * self.spent / self.budget
        return f"{self.spent}/{self.budget} units ({pct:.1f}%)"


def api_get(endpoint, params, quota, cost, max_retries=4):
    """GET an API endpoint with retries. Raises QuotaExhausted on 403 quota errors."""
    quota.charge(cost)
    params = dict(params)
    params["key"] = API_KEY
    url = f"{API_BASE}/{endpoint}"

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            return resp.json()

        # Distinguish "quota gone for the day" from "slow down"
        if resp.status_code == 403:
            body = resp.text.lower()
            if "quotaexceeded" in body or "dailylimitexceeded" in body:
                raise QuotaExhausted("Google reports daily quota exceeded.")
            if "ratelimitexceeded" in body or "userratelimit" in body:
                time.sleep(5 * (attempt + 1))
                continue

        if resp.status_code in (500, 503):
            time.sleep(2 ** attempt)
            continue

        # 404 / 400 — bad handle, deleted channel, etc. Caller decides.
        raise RuntimeError(f"{endpoint} -> HTTP {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError(f"{endpoint} failed after {max_retries} attempts")


# ---------------------------------------------------------------- duration

_ISO_DUR = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def parse_duration(iso):
    """'PT2M31S' -> 151 seconds. Returns None if unparseable (live streams)."""
    if not iso:
        return None
    m = _ISO_DUR.match(iso)
    if not m:
        return None
    parts = {k: int(v) for k, v in m.groupdict(default="0").items()}
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


# ---------------------------------------------------------------- channels

def load_handles(path="channels.txt"):
    handles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            handles.append(line.lstrip("@"))
    return handles


def resolve_channel(handle, quota):
    """handle -> dict with channel_id, uploads playlist, subscriber count."""
    data = api_get(
        "channels",
        {"part": "snippet,contentDetails,statistics", "forHandle": f"@{handle}"},
        quota,
        cost=1,
    )
    items = data.get("items", [])
    if not items:
        return None

    ch = items[0]
    stats = ch.get("statistics", {})

    # A channel can resolve but expose no uploads playlist (rare, but it
    # happens with brand accounts that have never published). Guard it here
    # so the caller can treat it as unresolved rather than crashing later.
    uploads = (
        ch.get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads")
    )
    if not uploads:
        return None

    return {
        "handle": handle,
        "channel_id": ch["id"],
        "channel_title": ch["snippet"]["title"],
        "channel_country": ch["snippet"].get("country", ""),
        "uploads_playlist": uploads,
        "subscriber_count": stats.get("subscriberCount"),
        "channel_video_count": stats.get("videoCount"),
        "subscribers_hidden": stats.get("hiddenSubscriberCount", False),
    }


# ---------------------------------------------------------------- videos

def fetch_upload_ids(playlist_id, quota, limit=MAX_VIDEOS_PER_CHANNEL):
    """Page through a channel's uploads playlist. 1 unit per 50 videos."""
    video_ids, page_token = [], None
    while len(video_ids) < limit:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token
        data = api_get("playlistItems", params, quota, cost=1)

        for item in data.get("items", []):
            vid = item.get("contentDetails", {}).get("videoId")
            if vid:
                video_ids.append(vid)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return video_ids[:limit]


def fetch_video_details(video_ids, quota):
    """Batch 50 IDs per call. 1 unit each."""
    out = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        data = api_get(
            "videos",
            {"part": "snippet,statistics,contentDetails,status", "id": ",".join(batch)},
            quota,
            cost=1,
        )
        out.extend(data.get("items", []))
    return out


def best_thumbnail(thumbs):
    """Prefer the largest available. hqdefault (480x360) is always present."""
    for key in ("maxres", "standard", "high", "medium", "default"):
        if key in thumbs:
            return thumbs[key]["url"]
    return ""


def flatten_video(video, channel):
    """One API video object -> one flat CSV row."""
    snip = video.get("snippet", {})
    stats = video.get("statistics", {})
    details = video.get("contentDetails", {})

    return {
        "video_id": video["id"],
        "channel_id": channel["channel_id"],
        "channel_handle": channel["handle"],
        "channel_title": channel["channel_title"],
        "channel_country": channel["channel_country"],
        "subscriber_count": channel["subscriber_count"],
        "published_at": snip.get("publishedAt", ""),
        "title": snip.get("title", ""),
        "description": snip.get("description", ""),
        "tags": "|".join(snip.get("tags", [])),
        "category_id": snip.get("categoryId", ""),
        "default_audio_language": snip.get("defaultAudioLanguage", ""),
        "live_broadcast_content": snip.get("liveBroadcastContent", "none"),
        "duration_seconds": parse_duration(details.get("duration")),
        "definition": details.get("definition", ""),
        "caption": details.get("caption", ""),
        "licensed_content": details.get("licensedContent", ""),
        # view_count can be missing when a channel hides statistics —
        # those rows are unusable and get dropped in the next step.
        "view_count": stats.get("viewCount"),
        "like_count": stats.get("likeCount"),
        "comment_count": stats.get("commentCount"),
        "thumbnail_url": best_thumbnail(snip.get("thumbnails", {})),
    }


# ---------------------------------------------------------------- pipeline

def collect(handles, quota):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resolved, unresolved = [], []

    for idx, handle in enumerate(handles, 1):
        cache_file = RAW_DIR / f"{handle}.json"

        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                resolved.append(payload["channel"])
                print(f"[{idx}/{len(handles)}] {handle}: cached, skipping")
                continue
            except (json.JSONDecodeError, KeyError):
                # Half-written cache from an interrupted run — redo it.
                print(f"[{idx}/{len(handles)}] {handle}: corrupt cache, refetching")
                cache_file.unlink()

        try:
            channel = resolve_channel(handle, quota)
        except QuotaExhausted:
            raise
        except Exception as e:
            print(f"[{idx}/{len(handles)}] {handle}: ERROR {e}")
            unresolved.append(handle)
            continue

        if channel is None:
            print(f"[{idx}/{len(handles)}] {handle}: HANDLE NOT FOUND")
            unresolved.append(handle)
            continue

        # A channel can resolve yet still fail here: its uploads playlist may
        # 404 (deleted/region-locked), or videos.list may reject something.
        # These are per-channel problems, not run-ending ones — log and skip,
        # exactly as we do for a handle that never resolved. Only QuotaExhausted
        # is allowed to stop the whole run.
        try:
            ids = fetch_upload_ids(channel["uploads_playlist"], quota)
            videos = fetch_video_details(ids, quota)
        except QuotaExhausted:
            raise
        except Exception as e:
            print(f"[{idx}/{len(handles)}] {handle}: FETCH FAILED {e}")
            unresolved.append(handle)
            continue

        # Write to a temp file then rename, so an interrupted run can never
        # leave a truncated JSON file that the resume logic would trust.
        # encoding="utf-8" is required — Windows defaults to cp1252, which
        # cannot encode the emoji that appear in plenty of video titles.
        tmp_file = cache_file.with_suffix(".tmp")
        tmp_file.write_text(
            json.dumps({"channel": channel, "videos": videos}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_file.replace(cache_file)
        resolved.append(channel)
        print(
            f"[{idx}/{len(handles)}] {handle}: {len(videos)} videos "
            f"| quota {quota.report()}"
        )

    return resolved, unresolved


def build_csv():
    """Flatten every cached channel file into one CSV."""
    rows = []
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unreadable cache file: {path.name}")
            continue
        channel = payload["channel"]
        for video in payload["videos"]:
            rows.append(flatten_video(video, channel))

    if not rows:
        print("No cached data found. Run the collector first.")
        return []

    DATA_DIR.mkdir(exist_ok=True)
    with open(VIDEOS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows):,} rows -> {VIDEOS_CSV}")
    return rows


def summarise(rows):
    """Diagnostics you need before deciding whether the dataset is usable."""
    from collections import Counter

    by_channel = Counter(r["channel_handle"] for r in rows)
    usable = [c for c, n in by_channel.items() if n >= 40]
    missing_views = sum(1 for r in rows if not r["view_count"])
    shorts = sum(1 for r in rows if r["duration_seconds"] and r["duration_seconds"] <= 60)
    live = sum(1 for r in rows if r["live_broadcast_content"] != "none")

    print("\n" + "=" * 55)
    print("DATASET SUMMARY")
    print("=" * 55)
    print(f"Total videos              : {len(rows):,}")
    print(f"Channels collected        : {len(by_channel)}")
    print(f"Channels with 40+ videos  : {len(usable)}   <- these are modellable")
    print(f"Rows missing view_count   : {missing_views:,}  (will be dropped)")
    print(f"Shorts (<=60s)            : {shorts:,} ({100*shorts/len(rows):.1f}%)")
    print(f"Live / upcoming           : {live:,}  (will be dropped)")

    thin = sorted((n, c) for c, n in by_channel.items() if n < 40)
    if thin:
        print(f"\nChannels too thin to use ({len(thin)}):")
        for n, c in thin[:15]:
            print(f"   {c}: {n} videos")


def download_thumbnails(limit=None):
    """hqdefault.jpg for every video. Costs no API quota. Resumable."""
    if not VIDEOS_CSV.exists():
        print("Run the metadata collection first.")
        return

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    with open(VIDEOS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    session = requests.Session()
    done = failed = skipped = 0

    for i, row in enumerate(rows, 1):
        vid = row["video_id"]
        out = THUMB_DIR / f"{vid}.jpg"
        if out.exists():
            skipped += 1
            continue

        url = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 1000:
                out.write_bytes(resp.content)
                done += 1
            else:
                failed += 1
        except requests.RequestException:
            failed += 1

        if i % 250 == 0:
            print(f"  {i}/{len(rows)} | new {done} | cached {skipped} | failed {failed}")
        time.sleep(0.03)   # be polite

    print(f"\nThumbnails: {done} downloaded, {skipped} already cached, {failed} failed")
    print(f"Saved to {THUMB_DIR}/")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true",
                    help="Only 3 channels — verify your key and setup first")
    ap.add_argument("--thumbnails", action="store_true",
                    help="Download thumbnail images (no API quota cost)")
    ap.add_argument("--channels", default="channels.txt")
    args = ap.parse_args()

    if args.thumbnails:
        download_thumbnails()
        return

    handles = load_handles(args.channels)
    if args.smoke_test:
        handles = handles[:3]
        print(f"SMOKE TEST: {handles}\n")

    quota = Quota()
    try:
        resolved, unresolved = collect(handles, quota)
    except QuotaExhausted as e:
        print(f"\n>>> QUOTA STOP: {e}")
        print(">>> Progress is cached. Re-run after midnight Pacific to resume.")
        resolved, unresolved = [], []

    if unresolved:
        print(f"\nCould not resolve {len(unresolved)} handles — fix or remove "
              f"them in {args.channels}:")
        for h in unresolved:
            print(f"   {h}")

    if resolved:
        DATA_DIR.mkdir(exist_ok=True)
        with open(CHANNELS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(resolved[0].keys()))
            writer.writeheader()
            writer.writerows(resolved)

    rows = build_csv()
    if rows:
        summarise(rows)
        print("\nNext: python collect_youtube.py --thumbnails")


if __name__ == "__main__":
    API_KEY = os.environ.get("YOUTUBE_API_KEY")
    if not API_KEY:
        sys.exit("Set YOUTUBE_API_KEY first.  export YOUTUBE_API_KEY='...'")
    main()
