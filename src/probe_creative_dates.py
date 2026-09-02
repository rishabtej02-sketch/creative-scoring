"""
Probe 5 saved creative pages to see what date/status info is actually
extractable from the per-creative detail view.

Reads source_href from existing scraped JSON, opens each creative page,
and prints:
  - any text matching 'last shown' / 'first shown' / date shapes
  - a dump of visible text blocks so we can see where the date lives
  - the raw DOM around any date, so we can write the real selector

Run headed so you can eyeball the page too:
  python src/probe_creative_dates.py --headed
"""

from __future__ import annotations
import argparse, glob, json, re, sys, time, random
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "google_ads"
BASE = "https://adstransparency.google.com"

DATE_HINT = re.compile(
    r"(first shown|last shown|shown|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2})",
    re.I,
)


def collect_hrefs(n: int) -> list[tuple[str, str]]:
    """Grab n (slug, source_href) pairs from across the scraped files."""
    out = []
    for f in sorted(glob.glob(str(OUT_DIR / "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for a in d.get("ads", []):
            href = a.get("source_href", "")
            if href:
                out.append((d.get("slug", Path(f).stem), href))
            if len(out) >= n:
                return out
    return out


def probe(headed: bool, n: int) -> None:
    pairs = collect_hrefs(n)
    if not pairs:
        sys.exit("no source_href found in any JSON")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"),
            viewport={"width": 1440, "height": 900}, locale="en-IN",
        )
        page = ctx.new_page()

        for i, (slug, href) in enumerate(pairs, 1):
            url = f"{BASE}{href}" if href.startswith("/") else href
            print("\n" + "=" * 70)
            print(f"[{i}/{len(pairs)}] {slug}")
            print(url)
            print("=" * 70)
            try:
                page.goto(url, wait_until="domcontentloaded")
            except Exception as e:
                print(f"  goto failed: {e!r}")
                continue

            # give the detail view time to render / lazy-load
            time.sleep(random.uniform(3.0, 5.0))
            # nudge in case content renders on interaction
            try:
                page.mouse.wheel(0, 800)
            except Exception:
                pass
            time.sleep(1.5)

            # 1) full visible text, line by line, filtered to date-ish lines
            try:
                body_text = page.inner_text("body")
            except Exception:
                body_text = ""
            date_lines = [ln.strip() for ln in body_text.splitlines()
                          if ln.strip() and DATE_HINT.search(ln)]
            print("  -- date-ish visible lines --")
            if date_lines:
                for ln in date_lines[:15]:
                    print(f"     | {ln}")
            else:
                print("     (none found in visible text)")

            # 2) show ALL short visible lines so we can see the layout
            #    (dates sometimes sit in their own tiny element with no keyword)
            short = [ln.strip() for ln in body_text.splitlines()
                     if 3 <= len(ln.strip()) <= 40]
            print("  -- all short visible lines (layout scan) --")
            for ln in short[:25]:
                print(f"     . {ln}")

            # 3) grep the raw HTML for date keywords + capture surrounding tag
            try:
                html = page.content()
            except Exception:
                html = ""
            hits = []
            for m in re.finditer(
                r".{0,60}(first shown|last shown).{0,80}", html, re.I
            ):
                hits.append(m.group(0))
            print("  -- raw HTML around 'shown' keywords --")
            if hits:
                for h in hits[:5]:
                    # collapse whitespace for readability
                    print("     >", re.sub(r"\s+", " ", h)[:160])
            else:
                print("     (keyword not present in HTML)")

        print("\n" + "=" * 70)
        print("done. if dates appear above -> enrichment is viable.")
        print("if 'none' everywhere -> Google isn't exposing it here.")
        if headed:
            input("press Enter to close browser...")
        ctx.close()
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("-n", type=int, default=5, help="how many creatives to probe")
    args = ap.parse_args()
    probe(args.headed, args.n)
