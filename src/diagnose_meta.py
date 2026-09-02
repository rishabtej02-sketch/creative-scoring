"""
diagnose_meta.py
================
Loads one brand's Meta Ad Library page, waits generously for ads to render,
then dumps everything I need to fix the scraper:
  - screenshot
  - first 20 KB of visible text
  - element counts for candidate selectors
  - outerHTML of a plausible "ads container"

Usage:
    python src/diagnose_meta.py 170229159669841

(Argument = Zomato page_id from your cache. Or any other page_id.)

Outputs to data/meta_ads/_diag/:
  - screenshot.png
  - body_text.txt
  - selector_counts.txt
  - first_card_html.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
USER_DATA_DIR = ROOT / ".playwright_meta"
OUT = ROOT / "data" / "meta_ads" / "_diag"

# Candidate selectors — I'll try a broad set and count matches
CANDIDATES = [
    "div[role='article']",
    "div._99s5",
    "div[data-testid='ad-item']",
    "div[aria-label*='ad' i]",
    "div._7jyi",
    "div._99s6",
    "div._7lu4",
    "div._8n_j",
    "div._9c1i",
    "div._8n86",
    "div[class^='x1']",
    "div._99s7",
    # generic containers that likely wrap each ad card
    "div[data-visualcompletion='ignore-dynamic']",
    "div[class*='ad_snapshot']",
    # library-id text nodes
    ":text('Library ID')",
    ":text-matches('Library ID', 'i')",
]

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/diagnose_meta.py <page_id>")
        return 1
    pid = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)

    url = (
        f"https://www.facebook.com/ads/library/"
        f"?active_status=all&ad_type=all&country=IN&view_all_page_id={pid}"
    )
    print("URL:", url)

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(USER_DATA_DIR),
            channel="chrome" if False else None,  # keep bundled chromium
            headless=False,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
        )
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        print("Loaded. Waiting 15s for ads to render...")
        page.wait_for_timeout(15_000)

        # scroll a bit to trigger lazy load
        for _ in range(3):
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(2000)

        # 1. screenshot
        page.screenshot(path=str(OUT / "screenshot.png"), full_page=False)
        print("→ screenshot.png")

        # 2. visible body text (first 20 KB)
        body_text = page.evaluate("() => document.body.innerText || ''")
        (OUT / "body_text.txt").write_text(body_text[:20000], encoding="utf-8")
        print(f"→ body_text.txt ({len(body_text)} chars total, first 20k saved)")

        # 3. selector counts
        counts = []
        for sel in CANDIDATES:
            try:
                n = page.locator(sel).count()
            except Exception as e:
                n = f"ERR: {e}"
            counts.append(f"{n!s:>6}  {sel}")
        (OUT / "selector_counts.txt").write_text("\n".join(counts), encoding="utf-8")
        print("→ selector_counts.txt")
        print("\nSelector counts:")
        print("\n".join(counts))

        # 4. HTML of first likely card — try to find "Library ID" text and dump its ancestor
        card_html = page.evaluate("""
        () => {
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          let node;
          while (node = walker.nextNode()) {
            if (/Library ID/i.test(node.nodeValue || '')) {
              // walk up 8 levels or until we find a "wide" container
              let el = node.parentElement;
              for (let i=0; i<8 && el; i++) {
                if (el.offsetWidth > 300 && el.offsetHeight > 200) return el.outerHTML.slice(0, 20000);
                el = el.parentElement;
              }
              return node.parentElement ? node.parentElement.outerHTML.slice(0, 20000) : null;
            }
          }
          return null;
        }
        """)
        if card_html:
            (OUT / "first_card_html.txt").write_text(card_html, encoding="utf-8")
            print("→ first_card_html.txt")
        else:
            (OUT / "first_card_html.txt").write_text("NO 'Library ID' TEXT FOUND", encoding="utf-8")
            print("!! No 'Library ID' text found on page — page may need login, or 0 ads exist, or region wrong")

        # 5. count 'Library ID' text occurrences (proxy for ad card count)
        n_lib = body_text.lower().count("library id")
        print(f"\nTotal 'Library ID' occurrences in body text: {n_lib}")

        print("\nInspect the browser window. Press Enter here to close.")
        try:
            input()
        except EOFError:
            time.sleep(5)
        ctx.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
