"""Hop-2 identity-signal extraction from candidate web pages (§3)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

_KNOWN_SOCIAL_DOMAINS = {
    "linkedin.com": "linkedin",
    "x.com": "x",
    "twitter.com": "x",
    "instagram.com": "instagram",
    "github.com": "github",
    "facebook.com": "facebook",
}

_SLUG_PATH_IGNORE = {
    "in", "profile", "user", "people", "posts", "photos",
    "status", "home", "about", "contact",
}


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_identity_signals(url: str, html: str) -> list[dict[str, Any]]:
    """Mine candidate page HTML for names, social handles, JSON-LD profiles, and sameAs links."""
    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_signal(sig_type: str, val: str, meta: dict[str, Any] | None = None) -> None:
        v = _clean_text(val)
        if not v or len(v) < 2 or len(v) > 80:
            return
        key = (sig_type, v.lower())
        if key in seen:
            return
        seen.add(key)
        item: dict[str, Any] = {"signal_type": sig_type, "value": v, "seed_page": url}
        if meta:
            item.update(meta)
        signals.append(item)

    # 1. URL slug signal
    parts = urlsplit(url)
    host = parts.netloc.lower()
    for domain, net_name in _KNOWN_SOCIAL_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            segments = [seg for seg in parts.path.split("/") if seg and seg.lower() not in _SLUG_PATH_IGNORE]
            if segments:
                add_signal("social_slug", segments[0], {"platform": net_name})

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return signals

    # 2. Title and OpenGraph / Twitter meta tags
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        t = _clean_text(title_tag.string)
        # Extract name before delimiter e.g. "Alice Goa | LinkedIn" -> "Alice Goa"
        name_cand = re.split(r"\s+[|\-–—•:]\s+", t)[0].strip()
        if name_cand and not any(p in name_cand.lower() for p in ("linkedin", "profile", "twitter", "home")):
            add_signal("title_name", name_cand)

    for prop in ("og:title", "twitter:title", "profile:first_name", "profile:last_name", "og:description"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            c = _clean_text(tag["content"])
            if prop in ("profile:first_name", "profile:last_name"):
                add_signal("profile_name", c)
            elif "title" in prop:
                cand = re.split(r"\s+[|\-–—•:]\s+", c)[0].strip()
                if cand and len(cand) < 50:
                    add_signal("og_title", cand)

    # 3. JSON-LD structured data (Person, ProfilePage, sameAs)
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            ld = json.loads(script.string)
            items = ld.get("@graph", [ld]) if isinstance(ld, dict) else (ld if isinstance(ld, list) else [])
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                etype = str(entry.get("@type", "")).lower()
                if "person" in etype or "profilepage" in etype:
                    if entry.get("name"):
                        add_signal("jsonld_name", entry["name"])
                    if entry.get("alternateName"):
                        add_signal("jsonld_alternate_name", entry["alternateName"])
                    same_as = entry.get("sameAs")
                    if isinstance(same_as, str):
                        add_signal("jsonld_sameas", same_as)
                    elif isinstance(same_as, list):
                        for s in same_as:
                            if isinstance(s, str):
                                add_signal("jsonld_sameas", s)
        except Exception:
            continue

    # 4. Links with rel="me" or rel="author"
    for a in soup.find_all("a", href=True):
        rel = a.get("rel", [])
        rel_str = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        if "me" in rel_str or "author" in rel_str:
            add_signal("rel_profile_link", a["href"])

    # 5. Visible @handles
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    visible_text = soup.get_text()
    for match in re.finditer(r"(?<!\w)@([A-Za-z0-9_]{3,25})(?!\w)", visible_text):
        add_signal("handle_mention", match.group(0))

    return signals
