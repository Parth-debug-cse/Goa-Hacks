"""SerpApi Bing search adapter (AH-6, AH-7)."""

from __future__ import annotations

import logging
import time
from typing import Any

from common.http_utils import create_session, response_meta
from pom.adapters.base import Candidate, SearchResponse
from pom.config import CONFIG
from pom.provenance import log_request

LOGGER = logging.getLogger(__name__)


class SerpApiBingAdapter:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or CONFIG.serpapi_key

    def search(self, query: str, timeout: float = 10.0, **kwargs: Any) -> SearchResponse:
        """Execute text search via SerpApi Bing engine (AH-7)."""
        if not self.api_key or not query.strip():
            return SearchResponse(engine="serpapi_bing", warning="bing_skipped: no api key or empty query")

        started = time.monotonic()
        params = {
            "engine": "bing",
            "api_key": self.api_key,
            "q": query,
            "count": "20",
        }
        try:
            session = create_session()
            resp = session.get("https://serpapi.com/search", params=params, timeout=max(0.5, timeout))
            status, size = response_meta(resp)
            prov_id = log_request("serpapi_bing", "GET", "https://serpapi.com/search",
                                  params, status, (time.monotonic() - started) * 1000, size,
                                  response_data=getattr(resp, "content", None))
            data = resp.json() if hasattr(resp, "json") else {}
            if not isinstance(data, dict):
                return SearchResponse(engine="serpapi_bing", provenance_id=prov_id, warning="unexpected_shape")

            candidates = []
            for item in (data.get("organic_results") or []):
                if not isinstance(item, dict):
                    continue
                link = item.get("link")
                if link:
                    candidates.append(Candidate(
                        url=link,
                        title=item.get("title"),
                        source_engine="bing_text",
                        match_confidence_hint="visual",
                        provenance_id=prov_id,
                    ))
            return SearchResponse(engine="serpapi_bing", candidates=candidates, provenance_id=prov_id, raw_payload=data)
        except Exception as err:
            LOGGER.warning("SerpApi Bing search error: %s", err)
            return SearchResponse(engine="serpapi_bing", warning=f"serpapi_bing_failed: {err}")
