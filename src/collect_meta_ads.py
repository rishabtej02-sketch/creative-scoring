"""
collect_meta_ads.py
===================

Scrape Meta Ad Library (facebook.com/ads/library) for a list of brands.

WHAT IT DOES
------------
For each brand in brands_ads.txt:
  1. Search Meta Ad Library for the brand name.
  2. Resolve the first matching Page ID.
  3. Load all ads for that Page in country=IN (or specified region),
     status=all (active + inactive), ad_type=all.
  4. Scroll to trigger lazy-loaded ad cards.
  5. Extract per ad: ad_archive_id, page_id, page_name, ad_creative_bodies (text),
     start_date, end_date (nullable = still running), platforms, media URLs
     (images + videos + thumbnails), impressions bucket if visible.
  6. Save one JSON file per brand under data/meta_ads/<brand_slug>.json
  7. Resumable — skips brands whose file already exists.

RESPONSIBLE SCRAPING
--------------------
- Uses a real Chromium browser via Playwright (JS-rendered, avoids DOM breakage).
- Persistent user-data dir → cookies survive between runs (fewer captchas).
- Throttle: --delay seconds between page navigations (default 6s).
- --smoke-test: process only first 3 brands, cap 20 ads each.
- Do NOT run this in parallel. Do NOT lower --delay below 3.

USAGE
-----
    # First-time setup
    pip install -r requirements_ads.txt
    playwright install chromium

    # Smoke test (3 brands, ~20 ads each)
    python src/collect_meta_ads.py --smoke-test

    # Full run (all brands)
    python src/collect_meta_ads.py

    # Resume a partial run
    python src/collect_meta_ads.py --resume

    # Headed mode (see browser, solve captcha manually if triggered)
    python src/collect_meta_ads.py --headed

OUTPUTS
-------
    data/meta_ads/<brand_slug>.json      one file per brand
    data/meta_ads/_page_ids.csv          brand -> page_id lookup cache
    data/meta_ads/_run_log.jsonl         per-brand run log (success/fail/count)

FIELDS PER AD (kept minimal, all safe/public)
---------------------------------------------
    ad_archive_id, page_id, page_name, ad_creative_bodies (list[str]),
    ad_delivery_start_time, ad_delivery_stop_time, is_active,
    publisher_platforms (list[str]), snapshot_url,
    creative_images (list[url]), creative_videos (list[url]),
    impressions_bucket (optional str), spend_bucket (optional str),
    scraped_at
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import Browser, BrowserContext, Page, TimeoutError as PWTimeout, sync_playwright

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BRANDS_FILE = ROOT / "brands_ads.txt"
OUT_DIR = ROOT / "data" / "meta_ads"
USER_DATA_DIR = ROOT / ".playwright_meta"  # persistent browser profile

BASE_URL = "https://www.facebook.com/ads/library"

DEFAULT_DELAY_S = 6
DEFAULT_MAX_SCROLLS = 40  # ~40 scrolls * ~10 cards = ~400 ads/brand hard cap
SMOKE_MAX_BRANDS = 3
SMOKE_MAX_SCROLLS = 2  # ~20 ads per brand in smoke

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("meta_ads")


# ---------------------------------------------------------------------------
# Brand list parsing
# ---------------------------------------------------------------------------

def parse_brands(path: Path) -> list[dict]:
    """Read brands_ads.txt → list of {'name', 'category', 'region'}."""
    brands = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            log.warning("Skip malformed line: %s", line)
            continue
        brands.append({"name": parts[0], "category": parts[1], "region": parts[2]})
    return brands


def slugify(name: str) -> str:
    """Zomato -> zomato,  boAt Lifestyle -> boat_lifestyle."""
    s = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[-\s]+", "_", s)


# ---------------------------------------------------------------------------
# Page-ID lookup cache
# ---------------------------------------------------------------------------

def load_page_id_cache() -> dict[str, str]:
    """Load brand_slug -> page_id from CSV cache."""
    cache_path = OUT_DIR / "_page_ids.csv"
    if not cache_path.exists():
        return {}
    out = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        slug, pid, name = (line.split(",", 2) + ["", ""])[:3]
        out[slug] = pid
    return out


def save_page_id_cache(cache: dict[str, str], names: dict[str, str]) -> None:
    """Persist brand_slug -> page_id."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "_page_ids.csv"
    lines = ["slug,page_id,resolved_name"]
    for slug, pid in sorted(cache.items()):
        nm = names.get(slug, "").replace(",", " ")
        lines.append(f"{slug},{pid},{nm}")
    cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Playwright helpers
# ---------------------------------------------------------------------------

def open_browser(playwright, headed: bool) -> BrowserContext:
    """Persistent context — cookies survive across runs."""
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ctx = playwright.chromium.launch_persistent_context(
        str(USER_DATA_DIR),
        headless=not headed,
        viewport={"width": 1440, "height": 900},
        locale="en-US",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        args=["--disable-blink-features=AutomationControlled"],
    )
    return ctx


def dismiss_cookie_banner(page: Page) -> None:
    """Meta shows a cookie banner on first visit — try to close it."""
    try:
        for text in ["Allow all cookies", "Only allow essential", "Decline optional"]:
            btn = page.get_by_role("button", name=re.compile(text, re.I))
            if btn.count() > 0:
                btn.first.click(timeout=2000)
                page.wait_for_timeout(500)
                return
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page-ID resolution
# ---------------------------------------------------------------------------

def resolve_page_id(page: Page, brand_name: str, region: str = "IN") -> tuple[str | None, str | None]:
    """
    Search Meta Ad Library for brand → return (page_id, resolved_page_name).

    Strategy: navigate to the search-typeahead URL and inspect network calls +
    DOM links of shape /ads/library/?...view_all_page_id=NNNN
    """
    search_url = (
        f"{BASE_URL}/?active_status=all&ad_type=all"
        f"&country={region}&search_type=page&media_type=all&q={brand_name}"
    )
    log.info("  resolving page id for %s ...", brand_name)
    try:
        page.goto(search_url, wait_until="domcontentloaded", timeout=30_000)
    except PWTimeout:
        log.warning("  goto timed out for %s", brand_name)
        return None, None

    dismiss_cookie_banner(page)
    page.wait_for_timeout(2500)

    # Look for links that contain view_all_page_id=<digits>
    hrefs = page.eval_on_selector_all(
        "a[href*='view_all_page_id=']",
        "els => els.map(e => e.getAttribute('href'))",
    )
    page_names = page.eval_on_selector_all(
        "a[href*='view_all_page_id='] div[dir='auto'], a[href*='view_all_page_id='] span",
        "els => els.slice(0,10).map(e => e.textContent)",
    )

    for href in hrefs:
        m = re.search(r"view_all_page_id=(\d+)", href or "")
        if m:
            pid = m.group(1)
            name = (page_names[0] if page_names else brand_name).strip()
            log.info("  → page_id=%s (%s)", pid, name)
            return pid, name

    log.warning("  no page id found for %s", brand_name)
    return None, None


# ---------------------------------------------------------------------------
# Ad extraction
# ---------------------------------------------------------------------------

def extract_ads_from_dom(page: Page) -> list[dict]:
    """
    Extract ad cards visible in DOM.

    STRATEGY (v2, 2026-08):
    Meta uses obfuscated CSS class names (x1..., x2...) that change constantly.
    Class-based selectors are useless. Instead we:
      1. Find all text nodes matching "Library ID: <digits>" (rock-solid anchor).
      2. Walk up DOM from each match to find the card container (a div wide
         enough + containing images).
      3. Parse card's innerText for dates, platforms, active status.
      4. Grab img/video src attributes from within the card DOM subtree.

    Handles both date formats:
      "24 Feb 2026"  (Indian / EU)
      "Feb 24, 2026" (US)
    """
    js = r"""
    () => {
      const results = [];
      const seen = new Set();

      // 1. Walk text nodes for "Library ID: <digits>"
      const walker = document.createTreeWalker(
        document.body, NodeFilter.SHOW_TEXT, null
      );
      const anchors = [];
      let node;
      while (node = walker.nextNode()) {
        const m = (node.nodeValue || '').match(/Library ID[:\s]+(\d{6,})/i);
        if (m) anchors.push({ id: m[1], node: node });
      }

      for (const { id, node } of anchors) {
        if (seen.has(id)) continue;
        seen.add(id);

        // 2. Walk up DOM to find enclosing card container
        //    heuristic: wide enough (> 350px) AND contains an image or video
        let card = node.parentElement;
        for (let i = 0; i < 15 && card; i++) {
          if (card.offsetWidth > 350 &&
              (card.querySelector('img[src*="fbcdn"], img[src*="scontent"], video'))) {
            break;
          }
          card = card.parentElement;
        }
        if (!card) card = node.parentElement;
        if (!card) continue;

        const txt = card.innerText || '';

        // 3. Dates — try Indian format first, then US
        let startMatch = txt.match(/Started running on\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})/);
        if (!startMatch) startMatch = txt.match(/Started running on\s+([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})/);
        let stopMatch  = txt.match(/Stopped running on\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})/);
        if (!stopMatch)  stopMatch  = txt.match(/Stopped running on\s+([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})/);

        // 4. Active flag — the word "Active" appears near top of each ad card
        //    "Inactive" would override
        const isActive = /\bActive\b/.test(txt) && !/\bInactive\b/i.test(txt);

        // 5. Platforms
        const platforms = [];
        ['Facebook','Instagram','Messenger','Audience Network','Threads','WhatsApp']
          .forEach(p => { if (txt.includes(p)) platforms.push(p); });

        // 6. Media inside the card
        const imgs = Array.from(card.querySelectorAll('img'))
          .map(i => i.src || '')
          .filter(s => s && !s.startsWith('data:') &&
                       (s.includes('scontent') || s.includes('fbcdn')));
        const vids = Array.from(card.querySelectorAll('video'))
          .map(v => v.src ||
                    (v.querySelector('source') && v.querySelector('source').src) || '')
          .filter(Boolean);

        // 7. Snapshot URL
        const anchor = card.querySelector('a[href*="ads/library"][href*="id="]');
        const snap = anchor ? anchor.href : null;

        // 8. Ad body copy — largest dir="auto" text block (heuristic)
        const bodyNodes = Array.from(card.querySelectorAll('div[dir="auto"], span[dir="auto"]'))
          .map(n => (n.innerText || '').trim())
          .filter(t => t && t.length > 20 && !t.startsWith('Library ID')
                   && !t.startsWith('Started running')
                   && !t.startsWith('Stopped running'));
        const body = bodyNodes.sort((a,b) => b.length - a.length)[0] || null;

        // 9. Impression & spend buckets (rare on commercial Indian ads, may still appear)
        const imp = txt.match(/Impressions[:\s]+([^\n]+)/i);
        const spend = txt.match(/Amount spent[:\s]+([^\n]+)/i);

        results.push({
          ad_archive_id: id,
          ad_creative_body: body,
          start_date_raw: startMatch ? startMatch[1] : null,
          stop_date_raw: stopMatch ? stopMatch[1] : null,
          is_active: isActive,
          platforms: platforms,
          impressions_bucket: imp ? imp[1].trim().slice(0, 100) : null,
          spend_bucket: spend ? spend[1].trim().slice(0, 100) : null,
          creative_images: [...new Set(imgs)],
          creative_videos: [...new Set(vids)],
          snapshot_url: snap,
        });
      }
      return results;
    }
    """
    try:
        return page.evaluate(js)
    except Exception as e:
        log.warning("  extract eval failed: %s", e)
        return []


def scroll_and_collect(page: Page, max_scrolls: int, delay_s: int) -> list[dict]:
    """Scroll down to load lazy ad cards, collect dedup by ad_archive_id."""
    seen: dict[str, dict] = {}
    prev_count = 0
    stagnant = 0
    for i in range(max_scrolls):
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(int(delay_s * 1000 / 2))  # half delay per scroll
        cards = extract_ads_from_dom(page)
        for c in cards:
            seen[c["ad_archive_id"]] = c
        cur = len(seen)
        log.info("    scroll %02d → %d ads total (+%d)", i + 1, cur, cur - prev_count)
        if cur == prev_count:
            stagnant += 1
            if stagnant >= 3:
                log.info("    no new ads for 3 scrolls → done")
                break
        else:
            stagnant = 0
        prev_count = cur
    return list(seen.values())


# ---------------------------------------------------------------------------
# Per-brand processing
# ---------------------------------------------------------------------------

def collect_brand(
    context: BrowserContext,
    brand: dict,
    page_id_cache: dict[str, str],
    resolved_names: dict[str, str],
    max_scrolls: int,
    delay_s: int,
) -> dict:
    """Resolve page_id (or use cache) → fetch ads → save JSON."""
    slug = slugify(brand["name"])
    out_path = OUT_DIR / f"{slug}.json"
    if out_path.exists():
        n = len(json.loads(out_path.read_text(encoding="utf-8")).get("ads", []))
        log.info("[skip] %s already collected (%d ads)", slug, n)
        return {"slug": slug, "status": "skip", "ads": n}

    page = context.new_page()
    try:
        # step 1: page id
        pid = page_id_cache.get(slug)
        name = brand["name"]
        if not pid:
            pid, resolved = resolve_page_id(page, brand["name"], brand["region"])
            if not pid:
                return {"slug": slug, "status": "no_page_id", "ads": 0}
            page_id_cache[slug] = pid
            resolved_names[slug] = resolved or brand["name"]
            save_page_id_cache(page_id_cache, resolved_names)
            name = resolved or brand["name"]
        else:
            log.info("[cache] %s → page_id=%s", slug, pid)

        # step 2: navigate to all ads for this page
        ads_url = (
            f"{BASE_URL}/?active_status=all&ad_type=all"
            f"&country={brand['region']}&view_all_page_id={pid}"
        )
        log.info("  fetching %s", ads_url)
        page.goto(ads_url, wait_until="domcontentloaded", timeout=45_000)
        dismiss_cookie_banner(page)
        page.wait_for_timeout(3000)

        # step 3: scroll and collect
        ads = scroll_and_collect(page, max_scrolls, delay_s)

        # step 4: normalize + save
        for a in ads:
            a["page_id"] = pid
            a["page_name"] = name
            a["brand_query"] = brand["name"]
            a["category"] = brand["category"]
            a["region"] = brand["region"]
            a["scraped_at"] = datetime.now(timezone.utc).isoformat()

        payload = {
            "brand": brand["name"],
            "slug": slug,
            "page_id": pid,
            "page_name": name,
            "category": brand["category"],
            "region": brand["region"],
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "count": len(ads),
            "ads": ads,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("[done] %s → %d ads saved to %s", slug, len(ads), out_path.name)
        return {"slug": slug, "status": "ok", "ads": len(ads)}
    finally:
        page.close()


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

def append_run_log(entry: dict) -> None:
    log_path = OUT_DIR / "_run_log.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**entry, "ts": datetime.now(timezone.utc).isoformat()}) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Meta Ad Library for brand list.")
    parser.add_argument("--brands", type=str, default=str(BRANDS_FILE))
    parser.add_argument("--smoke-test", action="store_true", help="First 3 brands, ~20 ads each")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--delay", type=int, default=DEFAULT_DELAY_S)
    parser.add_argument("--max-scrolls", type=int, default=None)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated brand slugs to process only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    brands = parse_brands(Path(args.brands))
    if not brands:
        log.error("No brands parsed from %s", args.brands)
        return 1

    if args.smoke_test:
        brands = brands[:SMOKE_MAX_BRANDS]
        max_scrolls = SMOKE_MAX_SCROLLS
        log.info("SMOKE TEST: %d brands, max_scrolls=%d", len(brands), max_scrolls)
    else:
        max_scrolls = args.max_scrolls or DEFAULT_MAX_SCROLLS

    if args.only:
        wanted = set(args.only.split(","))
        brands = [b for b in brands if slugify(b["name"]) in wanted]
        log.info("Filter --only → %d brands", len(brands))

    page_id_cache = load_page_id_cache()
    resolved_names = {slugify(b["name"]): b["name"] for b in brands}

    with sync_playwright() as pw:
        ctx = open_browser(pw, headed=args.headed)
        try:
            for i, brand in enumerate(brands, 1):
                log.info("─── %d/%d  %s (%s) ───", i, len(brands), brand["name"], brand["category"])
                try:
                    result = collect_brand(ctx, brand, page_id_cache, resolved_names, max_scrolls, args.delay)
                except Exception as e:
                    log.exception("[fail] %s: %s", brand["name"], e)
                    result = {"slug": slugify(brand["name"]), "status": "error", "ads": 0, "error": str(e)}
                append_run_log(result)
                # gentle pause between brands
                if i < len(brands):
                    time.sleep(args.delay)
        finally:
            ctx.close()

    log.info("All done. Outputs in %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
