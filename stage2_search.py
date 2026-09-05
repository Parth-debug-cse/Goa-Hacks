"""Stage 2: reverse-image search through SerpApi, Vision, and optional Bing."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from common.http_utils import create_session
from common.image_utils import compress_for_upload, extract_exif

LOGGER = logging.getLogger(__name__)
CandidateEngine = Literal[
    "serpapi_exact", "serpapi_visual", "google_vision",
    "bing_pages_including", "bing_visual_similar", "bing_text",
]


@dataclass
class CandidateURL:
    url: str
    title: str | None = None
    source_engine: CandidateEngine = "google_vision"
    thumbnail: str | None = None
    match_confidence_hint: Literal["exact", "visual"] = "visual"


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    tracking = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in tracking])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", query, ""))


def _json(response: Any) -> dict[str, Any]:
    try:
        status = getattr(response, "status_code", 200)
        if isinstance(status, int) and status >= 400:
            LOGGER.warning("HTTP provider response failed with status %s", status)
            return {}
        value = response.json()
        return value if isinstance(value, dict) else {}
    except (ValueError, AttributeError):
        return {}


def search_serpapi(image_bytes: bytes) -> list[CandidateURL]:
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        LOGGER.warning("serpapi_branch_skipped: no SERPAPI_API_KEY")
        return []
    try:
        session = create_session()
        upload = session.post(
            "https://serpapi.com/image",
            files={"image": ("source.jpg", image_bytes, "image/jpeg")},
            data={"api_key": key},
            timeout=10,
        )
        payload = _json(upload)
        image_id = payload.get("image_id")
        if not image_id:
            LOGGER.warning("SerpApi upload failed: %s", payload)
            return []
        results: list[CandidateURL] = []
        for kind, hint in (("exact_matches", "exact"), ("visual_matches", "visual")):
            response = session.get(
                "https://serpapi.com/search",
                params={"engine": "google_lens", "api_key": key, "image_id": image_id, "type": kind},
                timeout=10,
            )
            for item in _json(response).get(kind, []) or []:
                if not isinstance(item, dict) or not item.get("link"):
                    continue
                results.append(CandidateURL(
                    url=item["link"], title=item.get("title"), source_engine=f"serpapi_{'exact' if hint == 'exact' else 'visual'}",
                    thumbnail=item.get("thumbnail"), match_confidence_hint=hint,
                ))
        return results
    except Exception as error:  # provider isolation is intentional
        LOGGER.warning("SerpApi failed: %s", error)
        return []


def build_search_query(web_query: str, exif: dict[str, Any] | None = None) -> str:
    values = [web_query]
    if exif:
        for key in ("DateTimeOriginal", "Make", "Model"):
            value = exif.get(key)
            if value:
                values.append(str(value))
        gps = exif.get("GPSInfo")
        if gps:
            values.append(" ".join(str(item) for item in gps.values()) if isinstance(gps, dict) else str(gps))
    stop = {"photo", "image", "person", "stock", "stock photo", "unknown"}
    tokens: list[str] = []
    for value in values:
        for token in str(value).split():
            cleaned = token.strip(" ,;:|()[]{}")
            if cleaned and cleaned.lower() not in stop and cleaned.lower() not in {x.lower() for x in tokens}:
                tokens.append(cleaned)
    return " ".join(tokens)


def _vision_candidates(image_bytes: bytes) -> tuple[list[CandidateURL], str]:
    try:
        from google.cloud import vision
        response = vision.ImageAnnotatorClient().web_detection(image=vision.Image(content=image_bytes))
        web = getattr(response, "web_detection", None)
        if web is None:
            return [], ""
        results: list[CandidateURL] = []
        entities: list[tuple[float, str]] = []
        for entity in getattr(web, "web_entities", []) or []:
            description = getattr(entity, "description", None)
            if description:
                try:
                    score = float(getattr(entity, "score", 0) or 0)
                except (TypeError, ValueError):
                    score = 0.0
                entities.append((score, description))
        for page in getattr(web, "pages_with_matching_images", []) or []:
            url = getattr(page, "url", None)
            if url:
                results.append(CandidateURL(url=url, title=getattr(page, "page_title", None),
                                            source_engine="google_vision", match_confidence_hint="exact"))
        labels = [getattr(label, "label", None) for label in getattr(web, "best_guess_labels", []) or []]
        terms = [text for _, text in sorted(entities, reverse=True)[:3]] + [x for x in labels if x]
        stop = {"photo", "image", "person", "stock photo"}
        query = " ".join(dict.fromkeys(x.strip() for x in terms if x and x.lower() not in stop))
        return results, query
    except Exception as error:
        LOGGER.warning("Google Vision failed: %s", error)
        return [], ""


def search_google_vision(image_bytes: bytes) -> list[CandidateURL]:
    """Return direct pages found by Vision Web Detection."""
    return _vision_candidates(image_bytes)[0]


def search_bing_visual(image_bytes: bytes) -> list[CandidateURL]:
    key = os.environ.get("AZURE_BING_VISUAL_SEARCH_KEY")
    if not key:
        LOGGER.warning("bing_branch_skipped: no api key")
        return []
    endpoint = os.environ.get(
        "AZURE_BING_VISUAL_SEARCH_ENDPOINT",
        "https://api.bing.microsoft.com/v7.0/images/visualsearch",
    )
    try:
        response = create_session().post(
            endpoint,
            headers={"Ocp-Apim-Subscription-Key": key},
            files={"image": ("source.jpg", image_bytes, "image/jpeg")},
            timeout=10,
        )
        payload = _json(response)
        results: list[CandidateURL] = []
        for tag in payload.get("tags", []) or []:
            if not isinstance(tag, dict):
                continue
            for action in tag.get("actions", []) or []:
                if not isinstance(action, dict):
                    continue
                action_type = action.get("actionType")
                if action_type not in {"PagesIncluding", "VisualSearch"}:
                    continue
                values = (action.get("data") or {}).get("value", []) or []
                for item in values:
                    if not isinstance(item, dict) or not item.get("hostPageUrl"):
                        continue
                    results.append(CandidateURL(
                        url=item["hostPageUrl"], title=item.get("name"),
                        source_engine="bing_pages_including" if action_type == "PagesIncluding" else "bing_visual_similar",
                        thumbnail=item.get("thumbnailUrl"),
                        match_confidence_hint="exact" if action_type == "PagesIncluding" else "visual",
                    ))
        return results
    except Exception as error:
        LOGGER.warning("Bing Visual Search failed: %s", error)
        return []


def search_bing_text(query: str, timeout: float = 10) -> list[CandidateURL]:
    key = os.environ.get("AZURE_BING_VISUAL_SEARCH_KEY")
    if not key or not query.strip():
        return []
    endpoint = os.environ.get(
        "AZURE_BING_IMAGE_SEARCH_ENDPOINT",
        "https://api.bing.microsoft.com/v7.0/images/search",
    )
    try:
        response = create_session().get(
            endpoint,
            headers={"Ocp-Apim-Subscription-Key": key},
            params={"q": query, "count": 20, "safeSearch": "Moderate"},
            timeout=max(0.1, timeout),
        )
        payload = _json(response)
        results: list[CandidateURL] = []
        for item in payload.get("value", []) or []:
            if not isinstance(item, dict):
                continue
            url = item.get("hostPageUrl") or item.get("webSearchUrl")
            if url:
                results.append(CandidateURL(
                    url=url, title=item.get("name"), source_engine="bing_text",
                    thumbnail=item.get("thumbnailUrl"), match_confidence_hint="visual",
                ))
        return results
    except Exception as error:
        LOGGER.warning("Bing text search failed: %s", error)
        return []
def _filter_candidates(candidates: list[CandidateURL], query: str = "") -> list[CandidateURL]:
    blocked_domains = {
        "shutterstock.com", "istockphoto.com", "stockphoto.com",
        "amazon.com", "ebay.com", "aliexpress.com",
    }
    social_domains = {
        "linkedin.com", "x.com", "twitter.com", "instagram.com", "facebook.com",
    }

    def hostname(candidate: CandidateURL) -> str:
        return (urlsplit(candidate.url).hostname or "").lower().rstrip(".")

    def is_domain_or_subdomain(host: str, domain: str) -> bool:
        return host == domain or host.endswith(f".{domain}")

    def is_blocked(candidate: CandidateURL) -> bool:
        host = hostname(candidate)
        return any(is_domain_or_subdomain(host, domain) for domain in blocked_domains)

    def is_social(candidate: CandidateURL) -> bool:
        host = hostname(candidate)
        return any(is_domain_or_subdomain(host, domain) for domain in social_domains)

    filtered = [c for c in candidates if c.url and not is_blocked(c)]
    terms = {term.lower() for term in query.split() if len(term) > 2}
    def rank(candidate: CandidateURL) -> tuple[int, int, int, int]:
        exact = 0 if candidate.match_confidence_hint == "exact" else 1
        social_rank = 0 if is_social(candidate) else 1
        engine_rank = {"serpapi_exact": 0, "bing_pages_including": 1, "google_vision": 2,
                       "serpapi_visual": 3, "bing_visual_similar": 4, "bing_text": 5}.get(candidate.source_engine, 6)
        text = f"{candidate.title or ''} {candidate.url}".lower()
        relevance = 0 if terms and any(term in text for term in terms) else 1
        return exact, social_rank, engine_rank, relevance
    return sorted(filtered, key=rank)


def merge_candidates(*groups: list[CandidateURL]) -> list[CandidateURL]:
    merged: dict[str, CandidateURL] = {}
    for candidate in _filter_candidates([item for group in groups for item in group]):
        key = normalize_url(candidate.url)
        if key not in merged:
            merged[key] = candidate
    return _filter_candidates(list(merged.values()))


def process_search(image_path: str, timeout_seconds: float = 25.0) -> tuple[list[CandidateURL], list[str]]:
    started = time.monotonic()
    image_bytes = compress_for_upload(image_path)
    exif = extract_exif(image_path)
    warnings: list[str] = []
    groups: list[list[CandidateURL]] = []
    pool = ThreadPoolExecutor(max_workers=3)
    futures = [
        pool.submit(search_serpapi, image_bytes),
        pool.submit(_vision_candidates, image_bytes),
        pool.submit(search_bing_visual, image_bytes),
    ]
    query = ""
    try:
        completed = as_completed(futures, timeout=timeout_seconds)
        for future in completed:
            try:
                value = future.result()
                if isinstance(value, tuple):
                    groups.append(value[0])
                    query = build_search_query(value[1], exif)
                else:
                    groups.append(value)
            except Exception as error:
                warnings.append(f"search_branch_failed: {error}")
    except FuturesTimeoutError:
        warnings.append("search_timeout_budget_exceeded")
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    if exif:
        LOGGER.info("Extracted optional EXIF attributes: %s", sorted(exif))
    if not os.environ.get("AZURE_BING_VISUAL_SEARCH_KEY"):
        warnings.append("bing_branch_skipped: no api key")
    remaining = timeout_seconds - (time.monotonic() - started)
    if query and remaining > 0 and os.environ.get("AZURE_BING_VISUAL_SEARCH_KEY"):
        text_candidates = search_bing_text(query, timeout=min(10, remaining))
        groups.append(text_candidates)
    elif query and os.environ.get("AZURE_BING_VISUAL_SEARCH_KEY"):
        warnings.append("bing_text_fallback_skipped: timeout budget exhausted")
    return _filter_candidates(merge_candidates(*groups), query), warnings
