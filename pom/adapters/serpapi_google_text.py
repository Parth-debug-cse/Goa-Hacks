"""SerpApi Google plain web search adapter for hop-2 identity pivots (AH-6)."""

from __future__ import annotations

import logging
import time
from typing import Any

from common.http_utils import create_session, response_meta
from pom.adapters.base import Candidate, SearchResponse
from pom.config import CONFIG
from pom.provenance import log_request

LOGGER = logging.getLogger(__name__)


class SerpApiGoogleTextAdapter:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or CONFIG.serpapi_key

    def search(self, query: str, discovered_via: dict[str, Any] | None = None, **kwargs: Any) -> SearchResponse:
        """Search Google via SerpApi for targeted social queries (hop-2)."""
        if not self.api_key or not query.strip():
            return SearchResponse(engine="serpapi_hop2", warning="hop2_skipped: no api key or empty query")

        started = time.monotonic()
        params = {
            "engine": "google",
            "api_key": self.api_key,
            "q": query,
            "num": "10",
        }
        try:
            session = create_session()
            resp = session.get("https://serpapi.com/search", params=params, timeout=10)
            status, size = response_meta(resp)
            prov_id = log_request("serpapi_hop2", "GET", "https://serpapi.com/search",
                                  params, status, (time.monotonic() - started) * 1000, size,
                                  response_data=getattr(resp, "content", None))
            data = resp.json() if hasattr(resp, "json") else {}
            if not isinstance(data, dict):
                return SearchResponse(engine="serpapi_hop2", provenance_id=prov_id, warning="unexpected_shape")

            candidates = []
            for item in (data.get("organic_results") or []):
                if not isinstance(item, dict):
                    continue
                link = item.get("link")
                if link:
                    candidates.append(Candidate(
                        url=link,
                        title=item.get("title"),
                        source_engine="serpapi_hop2",
                        match_confidence_hint="visual",
                        provenance_id=prov_id,
                        search_hop=2,
                        discovered_via=discovered_via,
                    ))
            return SearchResponse(engine="serpapi_hop2", candidates=candidates, provenance_id=prov_id, raw_payload=data)
        except Exception as err:
            LOGGER.warning("SerpApi Google text search error: %s", err)
            return SearchResponse(engine="serpapi_hop2", warning=f"serpapi_google_failed: {err}")
