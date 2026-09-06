"""SerpApi Google Lens adapter (AH-6)."""

from __future__ import annotations

import logging
import time
from typing import Any

from common.http_utils import create_session, response_meta
from pom.adapters.base import Candidate, SearchResponse
from pom.config import CONFIG
from pom.provenance import log_request

LOGGER = logging.getLogger(__name__)


class SerpApiLensAdapter:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or CONFIG.serpapi_key

    def search(self, image_input: str | bytes, match_type: str = "exact_matches", **kwargs: Any) -> SearchResponse:
        """Search Google Lens via SerpApi image_id or public URL."""
        if not self.api_key:
            return SearchResponse(engine="serpapi_lens", warning="serpapi_skipped: no SERPAPI_API_KEY")

        session = create_session()
        image_id = None

        if isinstance(image_input, bytes):
            # Upload image bytes to SerpApi
            started = time.monotonic()
            try:
                upload = session.post(
                    "https://serpapi.com/image",
                    files={"image": ("source.jpg", image_input, "image/jpeg")},
                    data={"api_key": self.api_key},
                    timeout=10,
                )
                status, size = response_meta(upload)
                log_request("serpapi_image_upload", "POST", "https://serpapi.com/image",
                            {"api_key": self.api_key}, status, (time.monotonic() - started) * 1000, size,
                            response_data=getattr(upload, "content", None))
                payload = upload.json() if hasattr(upload, "json") else {}
                image_id = payload.get("image_id")
            except Exception as err:
                LOGGER.warning("SerpApi image upload error: %s", err)
                return SearchResponse(engine="serpapi_lens", warning=f"serpapi_upload_failed: {err}")
        else:
            image_id = str(image_input)

        if not image_id:
            return SearchResponse(engine="serpapi_lens", warning="serpapi_upload_failed: no image_id")

        started = time.monotonic()
        params = {
            "engine": "google_lens",
            "api_key": self.api_key,
            "image_id": image_id,
            "type": match_type,
        }
        engine_name = "serpapi_exact" if match_type == "exact_matches" else "serpapi_visual"

        try:
            resp = session.get("https://serpapi.com/search", params=params, timeout=10)
            status, size = response_meta(resp)
            prov_id = log_request(engine_name, "GET", "https://serpapi.com/search",
                                  params, status, (time.monotonic() - started) * 1000, size,
                                  response_data=getattr(resp, "content", None))
            
            data = resp.json() if hasattr(resp, "json") else {}
            if not isinstance(data, dict):
                LOGGER.warning("unexpected_shape engine=%s top_level_keys=[]", engine_name)
                return SearchResponse(engine=engine_name, provenance_id=prov_id, warning="unexpected_shape")

            matches = data.get(match_type, [])
            if matches is None and "exact_matches" not in data and "visual_matches" not in data:
                LOGGER.warning("unexpected_shape engine=%s top_level_keys=%s", engine_name, sorted(list(data.keys())))

            candidates = []
            for item in (matches or []):
                if not isinstance(item, dict):
                    continue
                link = item.get("link")
                if link:
                    candidates.append(Candidate(
                        url=link,
                        title=item.get("title"),
                        thumbnail=item.get("thumbnail"),
                        source_engine=engine_name,
                        match_confidence_hint="exact" if match_type == "exact_matches" else "visual",
                        provenance_id=prov_id,
                    ))
            return SearchResponse(engine=engine_name, candidates=candidates, provenance_id=prov_id, raw_payload=data)
        except Exception as err:
            LOGGER.warning("SerpApi Lens search error: %s", err)
            return SearchResponse(engine=engine_name, warning=f"serpapi_lens_failed: {err}")
