#!/usr/bin/env python
"""Capture the dashboard for docs and release notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="docs/assets/dashboard.png")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1050)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,
            color_scheme="light",
        )
        page.goto(args.url, wait_until="networkidle", timeout=60_000)
        page.wait_for_selector(".result-card", timeout=30_000)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(target), full_page=args.full)
        browser.close()
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
