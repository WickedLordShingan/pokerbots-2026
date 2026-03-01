#!/usr/bin/env python3
"""
Download match logs from the competition website by automating the download button clicks.

Usage:
  python download_logs.py [--matches-url URL] [--output-dir DIR] [--headless]

Environment variables (optional, for login):
  COMPETITION_EMAIL    - your login email
  COMPETITION_PASSWORD - your login password

If the match list is public, no login needed. If it requires login, set the env vars
and the script will attempt to log in before scraping.
"""
import argparse
import os
import re
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Install Playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


# Supabase public bucket base - logs are stored here
SUPABASE_LOG_BASE = "https://hzejxbyqialslmylqgbn.supabase.co/storage/v1/object/public/match_logs"
# UUID pattern for match IDs
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.I,
)


def extract_uuid_from_href(href: str) -> str | None:
    """Extract match UUID from href (e.g. '43c71ba7-0bfc-45a5-afaf-807cbdb71d00.log.gz' or full URL)."""
    if not href:
        return None
    m = UUID_PATTERN.search(href)
    return m.group(0) if m else None


def get_download_url(href: str, page_url: str) -> str:
    """Resolve full download URL from href (relative or absolute)."""
    href = href.strip()
    if href.startswith("http://") or href.startswith("https://"):
        return href
    # Relative: e.g. "43c71ba7-0bfc-45a5-afaf-807cbdb71d00.log.gz"
    if href.endswith(".log.gz") and UUID_PATTERN.search(href):
        uuid = extract_uuid_from_href(href)
        return f"{SUPABASE_LOG_BASE}/{uuid}.log.gz" if uuid else ""
    # Fallback: resolve against page origin
    from urllib.parse import urljoin, urlparse
    base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"
    return urljoin(base, href)


def main():
    parser = argparse.ArgumentParser(description="Download competition match logs")
    parser.add_argument(
        "--matches-url",
        default=os.environ.get("COMPETITION_MATCHES_URL", ""),
        help="URL of the match history page (or set COMPETITION_MATCHES_URL)",
    )
    parser.add_argument(
        "--output-dir",
        default="./logs/competition",
        help="Directory to save .log.gz files",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser headless",
    )
    parser.add_argument(
        "--login-url",
        default=os.environ.get("COMPETITION_LOGIN_URL", ""),
        help="Login page URL if matches require auth (or set COMPETITION_LOGIN_URL)",
    )
    parser.add_argument(
        "--click",
        action="store_true",
        help="Click each download button instead of fetching by URL (use if direct fetch fails)",
    )
    parser.add_argument(
        "--manual-login",
        action="store_true",
        help="Open browser and wait for you to log in manually (press Enter when ready)",
    )
    args = parser.parse_args()

    if not args.matches_url:
        print("Provide --matches-url or set COMPETITION_MATCHES_URL")
        print("Example: python download_logs.py --matches-url https://yoursite.com/matches")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    email = os.environ.get("COMPETITION_EMAIL")
    password = os.environ.get("COMPETITION_PASSWORD")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        # Login
        if args.manual_login:
            url = args.login_url or args.matches_url
            print(f"Opening {url} - log in manually, then press Enter here...")
            page.goto(url, wait_until="networkidle")
            input()
        elif email and password and args.login_url:
            print("Logging in...")
            page.goto(args.login_url, wait_until="networkidle")
            page.fill('input[type="email"], input[name="email"], input[id="email"]', email)
            page.fill('input[type="password"], input[name="password"], input[id="password"]', password)
            page.click('button[type="submit"], input[type="submit"], [type="submit"]')
            page.wait_for_load_state("networkidle")
            print("Logged in.")

        # Go to match list
        print(f"Opening {args.matches_url}...")
        page.goto(args.matches_url, wait_until="networkidle")

        # Find all download links
        links = page.query_selector_all('a[href*=".log.gz"]')
        hrefs = []
        for link in links:
            href = link.get_attribute("href")
            if href and ".log.gz" in href:
                hrefs.append(href)

        # Deduplicate by UUID
        seen = set()
        unique = []
        for href in hrefs:
            uuid = extract_uuid_from_href(href)
            if uuid and uuid not in seen:
                seen.add(uuid)
                unique.append((uuid, href))

        print(f"Found {len(unique)} match log(s).")

        if args.click:
            # Click each download button
            for uuid, href in unique:
                out_path = out_dir / f"{uuid}.log.gz"
                if out_path.exists():
                    print(f"  Skip (exists): {uuid}.log.gz")
                    continue
                try:
                    selector = f'a[href*="{uuid}"]'
                    link = page.query_selector(selector)
                    if not link:
                        print(f"  Skip (no link): {uuid}")
                        continue
                    with page.expect_download() as download_info:
                        link.click()
                    download = download_info.value
                    download.save_as(out_path)
                    print(f"  Downloaded: {uuid}.log.gz")
                except Exception as e:
                    print(f"  Error {uuid}: {e}")
        else:
            # Fetch by URL (works for public Supabase URLs)
            for uuid, href in unique:
                url = get_download_url(href, page.url)
                if not url:
                    print(f"  Skip (no URL): {href}")
                    continue

                out_path = out_dir / f"{uuid}.log.gz"
                if out_path.exists():
                    print(f"  Skip (exists): {uuid}.log.gz")
                    continue

                try:
                    resp = page.request.get(url)
                    if resp.ok:
                        out_path.write_bytes(resp.body())
                        print(f"  Downloaded: {uuid}.log.gz")
                    else:
                        print(f"  Failed ({resp.status}): {uuid}")
                except Exception as e:
                    print(f"  Error {uuid}: {e}")

        browser.close()

    print(f"Done. Logs saved to {out_dir.absolute()}")


if __name__ == "__main__":
    main()
