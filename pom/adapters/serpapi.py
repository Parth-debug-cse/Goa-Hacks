"""SerpApi adapter for Google Lens, Google Search, and Bing Search (AH-6).

Guarantees:
- AH-2: DEFENSIVE ACCESS ONLY. Logs `WARNING unexpected_shape engine=<x>` on unexpected structure.
- AH-7: Routes Bing through SerpApi engine=bing / engine=bing_images instead of retired direct endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlsplit

from common.http_utils import create_session, response_meta
from common.provenance import log_request

LOGGER = logging.getLogger(__name__)


def _log_unexpected_shape(engine: str, payload: dict[str, Any]) -> None:
    top_level_keys = sorted(list(payload.keys())) if isinstance(payload, dict) else []
    LOGGER.warning("unexpected_shape engine=%s top_level_keys=%s", engine, top_level_keys)


def upload_image_for_lens(image_bytes: bytes, api_key: str) -> str | None:
    """Upload local image bytes to SerpApi and return the assigned image_id."""
    session = create_session()
    started = time.monotonic()
    try:
        response = session.post(
            "https://serpapi.com/image",
            files={"image": ("source.jpg", image_bytes, "image/jpeg")},
            data={"api_key": api_key},
            timeout=10,
        )
        status, size = response_meta(response)
        log_request("serpapi_image_upload", "POST", "https://serpapi.com/image",
                    {"api_key": api_key}, status, (time.monotonic() - started) * 1000, size)
        
        if status is not None and status >= 400:
            LOGGER.warning("SerpApi image upload failed with status %s", status)
            return None
        
        data = response.json() if hasattr(response, "json") else {}
        if not isinstance(data, dict):
            _log_unexpected_shape("serpapi_image_upload", {})
            return None
        
        image_id = data.get("image_id")
        if not image_id:
            _log_unexpected_shape("serpapi_image_upload", data)
        return image_id
    except Exception as err:
        LOGGER.warning("SerpApi image upload exception: %s", err)
        return None


def search_google_lens(
    image_id: str,
    api_key: str,
    match_type: str = "exact_matches",
) -> tuple[list[dict[str, Any]], str]:
    """Query SerpApi Google Lens for exact or visual matches."""
    session = create_session()
    started = time.monotonic()
    params = {
        "engine": "google_lens",
        "api_key": api_key,
        "image_id": image_id,
        "type": match_type,
    }
    hint = "exact" if match_type == "exact_matches" else "visual"
    engine_name = f"serpapi_{hint}"
    
    try:
        response = session.get("https://serpapi.com/search", params=params, timeout=10)
        status, size = response_meta(response)
        prov_id = log_request(engine_name, "GET", "https://serpapi.com/search",
                              params, status, (time.monotonic() - started) * 1000, size)
        
        if status is not None and status >= 400:
            LOGGER.warning("SerpApi Lens %s returned HTTP %s", match_type, status)
            return [], prov_id

        data = response.json() if hasattr(response, "json") else {}
        if not isinstance(data, dict):
            _log_unexpected_shape(engine_name, {})
            return [], prov_id

        matches = data.get(match_type)
        if matches is None and "visual_matches" not in data and "exact_matches" not in data:
            _log_unexpected_shape(engine_name, data)

        results: list[dict[str, Any]] = []
        for item in (matches or []):
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if link:
                results.append({
                    "url": link,
                    "title": item.get("title"),
                    "thumbnail": item.get("thumbnail"),
                    "source": item.get("source"),
                    "match_hint": hint,
                })
        return results, prov_id
    except Exception as err:
        LOGGER.warning("SerpApi Lens query exception: %s", err)
        return [], ""


def search_serpapi_bing(
    query: str,
    api_key: str,
    timeout: float = 10.0,
) -> tuple[list[dict[str, Any]], str]:
    """Query Bing through SerpApi (AH-7: replaces retired direct Bing endpoint)."""
    session = create_session()
    started = time.monotonic()
    params = {
        "engine": "bing",
        "api_key": api_key,
        "q": query,
        "count": "20",
    }
    try:
        response = session.get("https://serpapi.com/search", params=params, timeout=max(0.5, timeout))
        status, size = response_meta(response)
        prov_id = log_request("serpapi_bing", "GET", "https://serpapi.com/search",
                              params, status, (time.monotonic() - started) * 1000, size)
        
        if status is not None and status >= 400:
            return [], prov_id

        data = response.json() if hasattr(response, "json") else {}
        if not isinstance(data, dict):
            _log_unexpected_shape("serpapi_bing", {})
            return [], prov_id

        results: list[dict[str, Any]] = []
        for item in (data.get("organic_results") or []):
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if link:
                results.append({
                    "url": link,
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                })
        return results, prov_id
    except Exception as err:
        LOGGER.warning("SerpApi Bing search exception: %s", err)
        return [], ""
