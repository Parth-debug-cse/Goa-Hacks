"""Candidate image URL extraction from HTML pages (§3)."""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup


def extract_image_urls(html: str, page_url: str) -> list[str]:
    """Extract candidate portrait/profile image URLs from page HTML."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    urls: list[str] = []
    # 1. High priority: OpenGraph and Twitter meta images
    for selector in ('meta[property="og:image"]', 'meta[name="twitter:image"]'):
        for tag in soup.select(selector):
            content = tag.get("content")
            if content:
                urls.append(urljoin(page_url, content))

    # 2. Filtered <img> tags
    for tag in soup.find_all("img"):
        src = tag.get("src")
        if not src:
            continue
        width = tag.get("width")
        height = tag.get("height")
        if (width and str(width).isdigit() and int(width) < 100) or (height and str(height).isdigit() and int(height) < 100):
            continue
        if any(token in src.lower() for token in ("logo", "icon", "sprite", "avatar-default", "tracker")):
            continue
        urls.append(urljoin(page_url, src))

    return list(dict.fromkeys(urls))
