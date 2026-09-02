"""
Google Ads Transparency Center scraper.
Mirrors collect_meta_ads.py: Playwright, resumable, throttled,
--smoke-test, --max-scrolls, --headed.

Output:
  data/google_ads/<brand_slug>.json
  data/google_ads/_advertiser_ids.csv   (slug -> AR-id cache)
  data/google_ads/_diag/                (screenshots + html when things break)

Site: https://adstransparency.google.com
No API. HTML lazy-loads on scroll. Class names are hashed -> we lean on
structural signals (links to /creative/CR..., <img>/<video> children,
date-shaped text) not brittle CSS classes.
"""

from __future__ import annotations
import argparse, csv, json, random, re, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from playwright.sync_api import sync_playwright, Page, Browser, TimeoutError as PWTimeout

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
BRANDS_FILE = ROOT / "brands_ads.txt"
OUT_DIR = ROOT / "data" / "google_ads"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ADV_CACHE = OUT_DIR / "_advertiser_ids.csv"
DIAG_DIR = OUT_DIR / "_diag"

BASE = "https://adstransparency.google.com"
DEFAULT_REGION = "IN"

# ---- polite throttle (match Meta scraper feel) ----
NAV_SLEEP = (2.5, 4.5)
SCROLL_SLEEP = (1.8, 3.2)
EARLY_STOP_ZERO_SCROLLS = 3


def slugify(name: str) -> str:
    return re.sub(r"[-\s]+", "_", name.lower()).strip("_")

def jitter(lo_hi): time.sleep(random.uniform(*lo_hi))


# ---- brand list ----
@dataclass
class Brand:
    name: str; slug: str; category: str; region: str

def load_brands(path: Path) -> list[Brand]:
    if not path.exists(): sys.exit(f"missing brand list: {path}")
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3: continue
        name, cat, region_field = parts[0], parts[1], parts[2]
        region = region_field.split("-")[0].strip() or DEFAULT_REGION
        out.append(Brand(name, slugify(name), cat, region))
    return out


# ---- advertiser id cache ----
def load_adv_cache() -> dict[str, str]:
    if not ADV_CACHE.exists(): return {}
    with ADV_CACHE.open(encoding="utf-8") as f:
        return {r["slug"]: r["advertiser_id"] for r in csv.DictReader(f)}

def save_adv_cache(cache: dict[str, str]) -> None:
    with ADV_CACHE.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "advertiser_id"]); w.writeheader()
        for k, v in sorted(cache.items()):
            w.writerow({"slug": k, "advertiser_id": v})


# ---- advertiser lookup ----
AR_URL_RE = re.compile(r"/advertiser/(AR\d+)")

def find_advertiser_id(page: Page, brand: Brand) -> str | None:
    page.goto(f"{BASE}/?region={brand.region}", wait_until="domcontentloaded")
    jitter(NAV_SLEEP)

    box = None
    for sel in ["input.input-area",
                "input[class*='input-area']",
                "input[aria-labelledby]",
                "input[aria-label*='Search' i]",
                "input[type='search']",
                "input[placeholder*='advertiser' i]",
                "input[type='text']"]:
        loc = page.locator(sel).first
        try:
            if loc.count() and loc.is_visible(): box = loc; break
        except Exception: continue
    if box is None: _dump(page, brand.slug, "no_search_box"); return None

    box.fill(""); box.type(brand.name, delay=40); jitter((1.5, 2.5))

    # Suggestions render as <div class="name">BRAND</div> (not anchors).
    # Wait for any suggestion div, then click the best match.
    try:
        page.wait_for_selector("div.name", timeout=8000)
    except PWTimeout:
        _dump(page, brand.slug, "no_suggestions"); return None

    items = page.locator("div.name")
    n = items.count()
    brand_lc = brand.name.lower()
    best_idx, best_score = None, -1
    for i in range(min(n, 20)):
        try: text = (items.nth(i).inner_text() or "").strip().lower()
        except Exception: text = ""
        if not text: continue
        score = (2 if brand_lc in text else 0) + (1 if text.startswith(brand_lc[:6]) else 0)
        if score > best_score:
            best_idx, best_score = i, score
    if best_idx is None:
        _dump(page, brand.slug, "no_match"); return None

    # Click, wait for navigation, read AR from URL.
    # Some suggestion divs don't trigger nav on child click -> try parent, then Enter.
    try:
        items.nth(best_idx).click()
    except Exception as e:
        print(f"  click failed: {e}")

    try:
        page.wait_for_url(re.compile(r"/advertiser/AR"), timeout=4000)
    except PWTimeout:
        # fallback 1: click parent row
        try:
            items.nth(best_idx).locator("xpath=..").click()
            page.wait_for_url(re.compile(r"/advertiser/AR"), timeout=3000)
        except Exception:
            # fallback 2: keyboard Enter
            try:
                page.keyboard.press("Enter")
                page.wait_for_url(re.compile(r"/advertiser/AR"), timeout=3000)
            except Exception:
                _dump(page, brand.slug, "no_nav"); return None

    m = AR_URL_RE.search(page.url)
    return m.group(1) if m else None


# ---- ad card extraction (in-page JS) ----
def extract_cards(page: Page) -> list[dict]:
    js = r"""
    () => {
      const out = []; const seen = new Set();
      const anchors = document.querySelectorAll("a[href*='/creative/CR']");
      for (const a of anchors) {
        const href = a.getAttribute('href') || '';
        const m = href.match(/\/creative\/(CR\d+)/); if (!m) continue;
        const cid = m[1]; if (seen.has(cid)) continue; seen.add(cid);

        let card = a;
        for (let i = 0; i < 6 && card && card.parentElement; i++) {
          card = card.parentElement;
          if (card.querySelector && card.querySelector('img, video')) break;
        }
        const img = card.querySelector('img');
        const vid = card.querySelector('video');
        const txt = (card.innerText || '').trim();
        const dateLine = (txt.split('\n').find(l =>
          /shown|20\d\d/i.test(l) &&
          /(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/i.test(l)
        )) || '';

        out.push({
          creative_id: cid, href,
          thumbnail_url: img ? (img.getAttribute('src') || '') : '',
          video_url: vid ? (vid.getAttribute('src') || '') : '',
          format: vid ? 'video' : (img ? 'image' : 'text'),
          card_text: txt.slice(0, 800),
          date_line: dateLine,
        });
      }
      return out;
    }"""
    try: return page.evaluate(js) or []
    except Exception: return []


def parse_date(s: str | None) -> str | None:
    if not s: return None
    s = s.strip()
    for fmt in ("%b %d, %Y", "%d %b %Y", "%Y-%m-%d"):
        try: return datetime.strptime(s, fmt).date().isoformat()
        except ValueError: continue
    return s

DATE_RANGE_RE  = re.compile(r"([A-Za-z]{3}\s+\d{1,2}(?:,\s*\d{4})?)\s*[\u2013-]\s*([A-Za-z]{3}\s+\d{1,2},\s*\d{4})")
FIRST_SHOWN_RE = re.compile(r"first shown[:\s]+([A-Za-z]{3}\s+\d{1,2},\s*\d{4})", re.I)
LAST_SHOWN_RE  = re.compile(r"last shown[:\s]+([A-Za-z]{3}\s+\d{1,2},\s*\d{4})",  re.I)

def enrich_card(card: dict, brand: Brand) -> dict:
    text = card.get("date_line", "") + " || " + card.get("card_text", "")
    first = last = None
    m = DATE_RANGE_RE.search(text)
    if m: first, last = parse_date(m.group(1)), parse_date(m.group(2))
    fm, lm = FIRST_SHOWN_RE.search(text), LAST_SHOWN_RE.search(text)
    if fm: first = parse_date(fm.group(1))
    if lm: last  = parse_date(lm.group(1))
    is_active = ("currently running" in text.lower()) or ("still running" in text.lower())
    return {
        "creative_id": card["creative_id"],
        "brand": brand.name, "brand_slug": brand.slug,
        "category": brand.category, "region": brand.region,
        "format": card["format"],
        "thumbnail_url": card.get("thumbnail_url", ""),
        "video_url": card.get("video_url", ""),
        "first_shown": first, "last_shown": last,
        "is_active": is_active,
        "ad_text": card.get("card_text", ""),
        "source_href": card.get("href", ""),
        "scrape_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---- scroll loop ----
def scroll_and_collect(page: Page, brand: Brand, max_scrolls: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for c in extract_cards(page): seen[c["creative_id"]] = c
    zero = 0
    for i in range(max_scrolls):
        prev = len(seen)
        page.mouse.wheel(0, 4000); jitter(SCROLL_SLEEP)
        try: page.keyboard.press("End")
        except Exception: pass
        jitter((0.4, 0.9))
        for c in extract_cards(page): seen.setdefault(c["creative_id"], c)
        new = len(seen) - prev
        print(f"  scroll {i+1:>2}: +{new:>3} (total {len(seen)})", flush=True)
        if new == 0:
            zero += 1
            if zero >= EARLY_STOP_ZERO_SCROLLS:
                print(f"  early stop: {EARLY_STOP_ZERO_SCROLLS} empty scrolls"); break
        else:
            zero = 0
    return [enrich_card(c, brand) for c in seen.values()]


# ---- diag dump ----
def _dump(page: Page, slug: str, tag: str) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = DIAG_DIR / f"{slug}_{tag}_{ts}"
    try:
        page.screenshot(path=str(stem) + ".png", full_page=True)
        Path(str(stem) + ".html").write_text(page.content(), encoding="utf-8")
        print(f"  diag -> {stem}.*")
    except Exception as e:
        print(f"  diag failed: {e}")


# ---- per-brand driver ----
def scrape_brand(browser: Browser, brand: Brand, max_scrolls: int,
                 adv_cache: dict[str, str], overwrite: bool) -> None:
    out_path = OUT_DIR / f"{brand.slug}.json"
    if out_path.exists() and not overwrite:
        print(f"[skip] {brand.slug}: exists -- pass --overwrite to redo"); return

    ctx = browser.new_context(
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
        viewport={"width": 1440, "height": 900}, locale="en-IN",
    )
    page = ctx.new_page()
    try:
        ar = adv_cache.get(brand.slug)
        if not ar:
            print(f"[lookup] {brand.slug} ...")
            ar = find_advertiser_id(page, brand)
            if not ar: print(f"[warn] no advertiser id for {brand.name} -- skip"); return
            adv_cache[brand.slug] = ar; save_adv_cache(adv_cache)

        url = f"{BASE}/advertiser/{ar}?region={brand.region}"
        print(f"[open] {brand.slug} -> {url}")
        page.goto(url, wait_until="domcontentloaded"); jitter(NAV_SLEEP)

        # Some advertiser pages need a scroll nudge before creatives lazy-load.
        found = False
        for attempt in range(3):
            try:
                page.wait_for_selector("a[href*='/creative/CR']", timeout=5000)
                found = True; break
            except PWTimeout:
                page.mouse.wheel(0, 2000); jitter((1.0, 2.0))
        if not found:
            _dump(page, brand.slug, "no_creatives")
            # Save empty payload so we don't retry this brand next run.
            (OUT_DIR / f"{brand.slug}.json").write_text(json.dumps({
                "brand": brand.name, "slug": brand.slug,
                "category": brand.category, "region": brand.region,
                "advertiser_id": ar,
                "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_ads": 0, "ads": [], "note": "no creatives visible in region",
            }, indent=2), encoding="utf-8")
            print(f"[warn] no creatives for {brand.name} -- wrote empty payload")
            return

        ads = scroll_and_collect(page, brand, max_scrolls)
        payload = {
            "brand": brand.name, "slug": brand.slug,
            "category": brand.category, "region": brand.region,
            "advertiser_id": ar,
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_ads": len(ads), "ads": ads,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] {brand.slug}: {len(ads)} ads -> {out_path.name}")
    finally:
        ctx.close()


# ---- main ----
def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brands-file", default=str(BRANDS_FILE))
    ap.add_argument("--only", nargs="*", help="brand slugs to include")
    ap.add_argument("--max-scrolls", type=int, default=30)
    ap.add_argument("--smoke-test", action="store_true",
                    help="first 3 brands, 5 scrolls each")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    brands = load_brands(Path(args.brands_file))
    if args.only:
        wanted = set(args.only)
        brands = [b for b in brands if b.slug in wanted]
    if args.smoke_test:
        brands = brands[:3]; args.max_scrolls = min(args.max_scrolls, 5)
    if not brands: sys.exit("no brands to scrape")

    adv_cache = load_adv_cache()
    print(f"brands: {len(brands)} | max_scrolls={args.max_scrolls} | headed={args.headed}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        try:
            for b in brands:
                try: scrape_brand(browser, b, args.max_scrolls, adv_cache, args.overwrite)
                except Exception as e: print(f"[err] {b.slug}: {e!r}")
                jitter(NAV_SLEEP)
        finally:
            browser.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
