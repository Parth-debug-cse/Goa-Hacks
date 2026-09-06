"""SerpApi Yandex visual search adapter (AH-6)."""

from __future__ import annotations

import logging
import time
from typing import Any

from common.http_utils import create_session, response_meta
from pom.adapters.base import Candidate, SearchResponse
from pom.config import CONFIG
from pom.provenance import log_request

LOGGER = logging.getLogger(__name__)


class SerpApiYandexAdapter:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or CONFIG.serpapi_key

    def search(self, image_url: str, **kwargs: Any) -> SearchResponse:
        """Search Yandex Reverse Image via SerpApi using a public image URL."""
        if not self.api_key:
            return SearchResponse(engine="serpapi_yandex", warning="yandex_skipped: no SERPAPI_API_KEY")

        started = time.monotonic()
        params = {
            "engine": "yandex_images",
            "api_key": self.api_key,
            "url": image_url,
        }
        try:
            session = create_session()
            resp = session.get("https://serpapi.com/search", params=params, timeout=10)
            status, size = response_meta(resp)
            prov_id = log_request("serpapi_yandex", "GET", "https://serpapi.com/search",
                                  params, status, (time.monotonic() - started) * 1000, size,
                                  response_data=getattr(resp, "content", None))
            data = resp.json() if hasattr(resp, "json") else {}
            if not isinstance(data, dict):
                return SearchResponse(engine="serpapi_yandex", provenance_id=prov_id, warning="unexpected_shape")

            candidates = []
            for item in (data.get("image_results") or []):
                if not isinstance(item, dict):
                    continue
                link = item.get("link")
                if link:
                    candidates.append(Candidate(
                        url=link,
                        title=item.get("title"),
                        thumbnail=item.get("thumbnail"),
                        source_engine="serpapi_yandex",
                        match_confidence_hint="visual",
                        provenance_id=prov_id,
                    ))
            return SearchResponse(engine="serpapi_yandex", candidates=candidates, provenance_id=prov_id, raw_payload=data)
        except Exception as err:
            LOGGER.warning("SerpApi Yandex search error: %s", err)
            return SearchResponse(engine="serpapi_yandex", warning=f"serpapi_yandex_failed: {err}")
