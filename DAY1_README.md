# Day 1 — Meta Ad Library data collection

Goal: pull real paid-ad creatives + run dates for 25-30 Indian D2C + global brands.

## Files added

- `brands_ads.txt` — brand list (edit freely; format `Name | category | region`)
- `requirements_ads.txt` — new Python deps (Playwright + tenacity)
- `src/collect_meta_ads.py` — the scraper

## Setup (one-time, ~2 min)

From `~/creative-scoring/`:

```bash
pip install -r requirements_ads.txt
playwright install chromium
```

## Smoke test (~5 min)

Runs 3 brands, ~20 ads each. Use this FIRST to verify things work before the full run.

```bash
python src/collect_meta_ads.py --smoke-test --headed
```

`--headed` opens a visible Chromium window so you can:
- see what's happening
- solve a CAPTCHA manually if Meta throws one (rare, first-run only)
- confirm results look sane

Expected output:
```
data/meta_ads/zomato.json
data/meta_ads/swiggy.json
data/meta_ads/licious.json
data/meta_ads/_page_ids.csv
data/meta_ads/_run_log.jsonl
```

Open one of the JSONs and eyeball:
- `count` > 5? Good.
- `ads[0]` has `ad_archive_id`, `start_date_raw`, `is_active`, `creative_images` or `creative_videos`? Good.
- All fields null? Something broke — Meta likely changed markup. Ping me.

## Full run (~2-4 hours, unattended)

Once smoke test passes:

```bash
python src/collect_meta_ads.py
```

Runs all 30 brands at ~6s throttle. Resumable — re-running skips brands
whose JSON already exists.

To force one brand only (e.g. re-scrape Zomato):

```bash
rm data/meta_ads/zomato.json
python src/collect_meta_ads.py --only zomato
```

## What we get

Per ad:
- `ad_archive_id` — unique Meta ID (primary key)
- `start_date_raw` + `stop_date_raw` — for **duration proxy label**
- `is_active` — still running vs killed
- `platforms` — FB / IG / Messenger / Audience Network
- `creative_images` + `creative_videos` — download URLs (Day 2 handles download)
- `ad_creative_body` — the ad text
- `impressions_bucket` + `spend_bucket` — new 2026 fields (mostly empty for
  commercial Indian ads; will still capture when present)

## Realistic expectations

| brand | typical ad count |
|---|---|
| Zomato, Swiggy, Meesho, Amazon India | 100-500+ |
| Nykaa, Mamaearth, boAt, CRED | 50-200 |
| Smaller D2C (SUGAR, Bewakoof, Purplle) | 20-100 |
| Fintech (Groww, Zerodha) | 10-80 |

Target: **~2000-4000 total ads across all brands.** Enough for ML.

## Known risks

- **Meta may throw a CAPTCHA** on first visit → `--headed` lets you solve it once,
  cookie persists in `.playwright_meta/` for future runs.
- **Markup drift**: Meta redesigns Ad Library every few months. If ad counts
  come back zero across brands, the DOM selectors need updating (function
  `extract_ads_from_dom` in the scraper).
- **Rate limits**: If Meta throttles you (blank pages), increase `--delay` to
  10-15 seconds and retry.
- **India-only ads**: brands with only foreign ads (e.g. Ryanair) will return
  zero for `region=IN`. Change `region` in `brands_ads.txt` to `GB` or `US`
  for those.

## After Day 1 completes

Report back:
1. How many brands returned ≥ 20 ads
2. Rough total ad count
3. Any brands that failed

Then Day 2 = creative download + Google Transparency scraper.
