"""Stage 2 — hop-2 identity pivot.

Hop-1 reverse-image search frequently returns visually-similar pages (stock
photos, unrelated content) instead of the person's actual social profile. Hop-2
closes that gap: :func:`extract_identity_signals` pulls identity signals out of
the pages hop-1 found (titles, JSON-LD Person/ProfilePage objects with
``sameAs`` profile lists, author/rel links, @handles, and social slugs already
present in the URLs), and :func:`run_hop2_search` turns those into a small
batch of targeted, site-scoped web queries that are run through SerpApi's plain
``google`` engine.

The hop-2 result is a list of :class:`stage2_search.CandidateURL` objects whose
``source_engine`` is ``"serpapi_hop2"``, carrying a ``discovered_via`` dict that
records the exact seed page + signal + query that produced them.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator
from urllib.parse import urlsplit

from common.http_utils import create_session, response_meta
from common.netguard import UnsafeURLError, assert_public_url
from common.provenance import log_request
from stage2_search import CandidateURL, _json

LOGGER = logging.getLogger(__name__)

# Tokens that add no signal to a name-built web query.
_NAME_STOPWORDS = {
    "the", "and", "or", "of", "for", "in", "on", "at", "to", "a", "an",
    "is", "are", "was", "were", "with", "by", "from", "it", "its", "his",
    "her", "this", "that", "not", "as", "be", "but", "if", "we", "you",
    "they", "them", "she", "he", "photo", "image", "picture", "portrait",
}

_TITLE_SEPARATORS = re.compile(r"\s*[|\-—·:]\s*")
_TITLE_CASE_TOKEN = re.compile(r"^[A-Z][A-Za-z']{1,30}$")
_MENTION_TAG = re.compile(r"\B@[A-Za-z0-9_]{2,30}\b")

# Known-chrome path segments that are not real profile slugs.
_NON_SLUG_SEGMENTS = {"home", "search", "status", "login", "intent", "share", "hashtag"}

# Social slug patterns; the signal_type is suffixed with the platform so the
# query builder knows which site to scope against.
_PROFILE_SLUG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"linkedin\.com/in/([A-Za-z0-9\-_]+)"), "linkedin"),
    (re.compile(r"(?:x\.com|twitter\.com)/([A-Za-z0-9_]{2,30})(?:[/?#]|$)"), "twitter"),
    (re.compile(r"instagram\.com/([A-Za-z0-9_.]{2,30})(?:[/?#]|$)"), "instagram"),
    (re.compile(r"github\.com/([A-Za-z0-9\-_]+)(?:[/?#]|$)"), "github"),
    (re.compile(r"facebook\.com/([A-Za-z0-9.\-]+)(?:[/?#]|$)"), "facebook"),
)


def _signal(signal_type: str, value: str, seed_page: str) -> dict[str, str]:
    return {"signal_type": signal_type, "value": str(value), "seed_page": seed_page}


# --------------------------------------------------------------------------- #
# Signal extraction
# --------------------------------------------------------------------------- #
def extract_identity_signals(page_url: str, page_html: str) -> list[dict]:
    """Extract identity signals from a fetched page's HTML.

    Returns a list of ``{"signal_type", "value", "seed_page"}`` dicts covering
    titles/OG tags, JSON-LD Person/ProfilePage objects, author + rel links,
    @handles in visible text, and social slugs present in ``page_url`` itself.
    """
    signals: list[dict] = list(_title_signals(page_html, page_url))
    signals += list(_jsonld_signals(page_html, page_url))
    signals += list(_author_link_signals(page_html, page_url))
    signals += list(_mention_signals(page_html, page_url))
    signals += list(_url_slug_signals(page_url))
    site_name = next(
        (s["value"] for s in signals if s["signal_type"] == "og:site_name"), None
    )
    signals += list(_title_name_signals(page_html, page_url, site_name))
    return signals


def _squash(text: str) -> str:
    return " ".join(str(text).split())


def _iter_meta_content(html: str, attribute: str) -> Iterator[str]:
    """Yield ``content`` values from ``<meta ...>`` tags with the given attr."""
    for tag in re.finditer(r"<meta\b[^>]*>", html, re.IGNORECASE):
        text = tag.group(0)
        if not re.search(attribute, text, re.IGNORECASE):
            continue
        match = re.search(r'content\s*=\s*["\'](.*?)["\']', text, re.IGNORECASE)
        if match:
            yield match.group(1)


def _title_signals(html: str, page_url: str) -> Iterator[dict]:
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title and _squash(title.group(1)):
        yield _signal("title", _squash(title.group(1)), page_url)

    meta_rules = (
        ("og:title", r'property\s*=\s*["\']og:title["\']'),
        ("og:site_name", r'property\s*=\s*["\']og:site_name["\']'),
        ("og:description", r'property\s*=\s*["\']og:description["\']'),
        ("og:description", r'name\s*=\s*["\']description["\']'),
    )
    for signal_type, attribute in meta_rules:
        for value in _iter_meta_content(html, attribute):
            if value:
                yield _signal(signal_type, value, page_url)


def _jsonld_signals(html: str, page_url: str) -> Iterator[dict]:
    for block in re.finditer(
        r'<script\b[^>]*type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        raw = block.group(1).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        visited: list[dict] = _flatten_jsonld(data)
        for obj in visited:
            kind = obj.get("@type")
            kinds = set(kind if isinstance(kind, list) else [kind])
            if not kinds.intersection({"Person", "ProfilePage"}):
                continue
            for field, signal_type in (
                ("name", "jsonld_name"),
                ("alternateName", "jsonld_alternate_name"),
            ):
                value = obj.get(field)
                if value:
                    yield _signal(signal_type, _squash(value), page_url)
            same_as = obj.get("sameAs")
            if isinstance(same_as, str):
                same_as = [same_as]
            for link in same_as or []:
                if isinstance(link, str) and link.strip():
                    yield _signal("sameAs", link.strip(), page_url)


def _flatten_jsonld(data: Any) -> list[dict]:
    """Collect the top-level object(s) plus any ``@graph`` descendants."""
    stack: list[Any] = data if isinstance(data, list) else [data]
    visited: list[dict] = []
    while stack:
        obj = stack.pop()
        if not isinstance(obj, dict):
            continue
        visited.append(obj)
        graph = obj.get("@graph")
        if isinstance(graph, list):
            stack.extend(graph)
    return visited


def _author_link_signals(html: str, page_url: str) -> Iterator[dict]:
    for value in _iter_meta_content(html, r'name\s*=\s*["\']author["\']'):
        if value:
            yield _signal("meta_author", value, page_url)

    for tag in re.finditer(r"<(?:a|link)\b[^>]*>", html, re.IGNORECASE):
        text = tag.group(0)
        if not re.search(r'\brel\s*=\s*["\'][^"\']*(?:author|me)[^"\']*["\']', text, re.IGNORECASE):
            continue
        href = re.search(r'\bhref\s*=\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        if href:
            yield _signal("rel_profile", href.group(1), page_url)


def _visible_text(html: str) -> str:
    text = re.sub(
        r"<script\b[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL
    )
    return re.sub(r"<[^>]+>", " ", text)


def _mention_signals(html: str, page_url: str) -> Iterator[dict]:
    for match in _MENTION_TAG.finditer(_visible_text(html)):
        handle = match.group(0)[1:]
        if handle:
            yield _signal("handle", handle, page_url)


def _url_slug_signals(page_url: str) -> Iterator[dict]:
    parts = urlsplit(page_url)
    candidate = (parts.hostname or "") + parts.path
    for pattern, platform in _PROFILE_SLUG_PATTERNS:
        match = pattern.search(candidate)
        if not match:
            continue
        slug = match.group(1)
        if slug.lower() in _NON_SLUG_SEGMENTS:
            continue
        yield _signal(f"url_slug:{platform}", slug, page_url)


def _title_name_signals(
    html: str, page_url: str, site_name: str | None = None
) -> Iterator[dict]:
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not title:
        return
    site_tokens = {token.lower().rstrip(".,") for token in (site_name or "").split()}
    for segment in _TITLE_SEPARATORS.split(_squash(title.group(1))):
        token_values = [value for value in segment.split() if value]
        if not 2 <= len(token_values) <= 4:
            continue
        if not all(_TITLE_CASE_TOKEN.match(value) for value in token_values):
            continue
        lowered = {value.lower().rstrip(".,") for value in token_values}
        # Only drop a segment that *is* the site name verbatim, so real names
        # that merely share a word with the site still survive.
        if site_tokens and lowered == site_tokens:
            continue
        if len(lowered) == 1 or lowered.issubset(_NAME_STOPWORDS):
            continue
        yield _signal("title_name", " ".join(token_values), page_url)


# --------------------------------------------------------------------------- #
# Hop-2 search
# --------------------------------------------------------------------------- #
def _is_plain_name(value: str) -> bool:
    tokens = [token for token in value.split() if token]
    if not tokens or len(tokens) > 4:
        return False
    lowered = {token.lower().rstrip(".,") for token in tokens}
    return not lowered.issubset(_NAME_STOPWORDS)


def _handle_from_social_url(url: str) -> Iterator[str]:
    parts = urlsplit(url)
    candidate = (parts.hostname or "") + parts.path
    for pattern, platform in _PROFILE_SLUG_PATTERNS:
        match = pattern.search(candidate)
        if match and match.group(1).lower() not in _NON_SLUG_SEGMENTS:
            yield match.group(1)
            return


_PLATFORM_SITES = {
    "linkedin": "linkedin.com/in",
    "twitter": "x.com",
    "instagram": "instagram.com",
    "github": "github.com",
    "facebook": "facebook.com",
}


def _build_hop2_queries(signals: list[dict]) -> list[tuple[str, dict]]:
    """Return ordered ``(query, discovered_via)`` pairs, deduplicated."""
    name_signals: list[dict] = []
    handle_signals: list[dict] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        signal_type = signal.get("signal_type")
        value = signal.get("value")
        seed_page = signal.get("seed_page", "")
        if signal_type in {"jsonld_name", "jsonld_alternate_name", "meta_author", "title_name"}:
            if _is_plain_name(str(value)):
                name_signals.append(signal)
        elif signal_type == "handle" and value:
            handle_signals.append(signal)
        elif signal_type and signal_type.startswith("url_slug:") and value:
            handle_signals.append(signal)
        elif signal_type == "sameAs" and value:
            for handle in _handle_from_social_url(str(value)):
                handle_signals.append({"signal_type": "handle", "value": handle, "seed_page": seed_page})

    queries: list[tuple[str, dict]] = []
    seen: set[str] = set()

    for signal in name_signals:
        for site in ("linkedin.com/in", "x.com", "instagram.com"):
            query = f'"{signal["value"]}" site:{site}'
            if query in seen:
                continue
            seen.add(query)
            queries.append((query, {
                "seed_page": signal["seed_page"], "signal_type": signal["signal_type"], "query": query,
            }))

    for signal in handle_signals:
        value = str(signal["value"]).lstrip("@")
        if not value or value.lower() in _NON_SLUG_SEGMENTS:
            continue
        signal_type = signal["signal_type"]
        if signal_type.startswith("url_slug:"):
            platform = signal_type.split(":", 1)[1]
            site = _PLATFORM_SITES.get(platform)
        else:
            site = None
        if site:
            query = f'"@{value}" site:x.com' if site == "x.com" else f'"{value}" site:{site}'
        else:
            query = f'"@{value}" site:x.com'
        if query in seen:
            continue
        seen.add(query)
        queries.append((query, {
            "seed_page": signal["seed_page"],
            "signal_type": signal.get("signal_type", "handle"),
            "query": query,
        }))
    return queries


def run_hop2_search(
    signals: list[dict], serpapi_key: str, max_queries: int = 6
) -> list[CandidateURL]:
    """Run targeted SerpApi ``google`` queries built from identity signals.

    Args:
        signals: output of :func:`extract_identity_signals` (one or more pages).
        serpapi_key: SerpApi API key.
        max_queries: hard cap on the number of search queries issued.

    Returns:
        Hop-2 candidates with ``source_engine="serpapi_hop2"``, ``search_hop=2``,
        a provenance id, and a ``discovered_via`` dict.
    """
    if not serpapi_key:
        LOGGER.warning("hop2_search_skipped: no serpapi key")
        return []
    queries = _build_hop2_queries(signals)[:max_queries]
    if not queries:
        LOGGER.info("hop2_search_skipped: no identity signals produced queries")
        return []

    results: list[CandidateURL] = []
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
        futures = {
            pool.submit(_serpapi_google_query, query, discovered_via, serpapi_key): discovered_via
            for query, discovered_via in queries
        }
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as error:  # provider isolation is intentional
                LOGGER.warning("hop2 query failed: %s", error)
    return results


def _serpapi_google_query(query: str, discovered_via: dict, key: str) -> list[CandidateURL]:
    """One plain-``google``-engine SerpApi query; parsed into candidates."""
    params = {"engine": "google", "api_key": key, "q": query}
    started = time.monotonic()
    try:
        response = create_session().get(
            "https://serpapi.com/search", params=params, timeout=10
        )
        status, size = response_meta(response)
        provenance_id = log_request(
            "serpapi_hop2", "GET", "https://serpapi.com/search", params,
            status, (time.monotonic() - started) * 1000, size,
        )
        candidates: list[CandidateURL] = []
        for item in _json(response).get("organic_results", []) or []:
            if not isinstance(item, dict) or not item.get("link"):
                continue
            candidates.append(CandidateURL(
                url=item["link"],
                title=item.get("title"),
                source_engine="serpapi_hop2",
                match_confidence_hint="exact",
                provenance_id=provenance_id,
                search_hop=2,
                discovered_via=dict(discovered_via),
            ))
        return candidates
    except Exception as error:
        log_request(
            "serpapi_hop2", "GET", "https://serpapi.com/search", params,
            None, (time.monotonic() - started) * 1000, None, str(error),
        )
        LOGGER.warning("SerpApi hop-2 query failed: %s", error)
        return []


def fetch_page_html(
    page_url: str, max_chars: int = 400_000, timeout: float = 8.0
) -> tuple[str | None, str | None]:
    """Fetch a candidate page and return ``(html, provenance_id)``.

    Only ``text/html`` pages under ``max_chars`` are returned; anything else
    yields ``(None, provenance_id)``. The URL is checked with the SSRF guard
    before any request is made.
    """
    try:
        assert_public_url(page_url)
    except UnsafeURLError as error:
        LOGGER.warning("Skipping unsafe hop-2 seed page %s: %s", page_url, error)
        return None, None

    started = time.monotonic()
    try:
        response = create_session().get(page_url, timeout=timeout)
    except Exception as error:
        log_request(
            "hop2_page_fetch", "GET", page_url, {},
            None, (time.monotonic() - started) * 1000, None, str(error),
        )
        return None, None
    status, size = response_meta(response)
    provenance_id = log_request(
        "hop2_page_fetch", "GET", page_url, {},
        status, (time.monotonic() - started) * 1000, size,
    )
    if status is not None and status >= 400:
        return None, provenance_id
    content_type = (response.headers.get("Content-Type") or "").lower()
    if not content_type.startswith("text/html"):
        return None, provenance_id
    return response.text[:max_chars], provenance_id