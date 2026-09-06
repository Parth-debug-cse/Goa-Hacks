"""Google Cloud Vision Web Detection adapter (AH-6)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from pom.adapters.base import Candidate, SearchResponse
from pom.config import CONFIG
from pom.provenance import log_request

LOGGER = logging.getLogger(__name__)


class GCVWebAdapter:
    def __init__(self, credentials_path: str | None = None) -> None:
        self.credentials_path = credentials_path or CONFIG.gcv_credentials

    def search(self, image_bytes: bytes, **kwargs: Any) -> SearchResponse:
        """Call Google Cloud Vision web detection and return pages & attributes."""
        try:
            from google.cloud import vision
        except ImportError:
            return SearchResponse(engine="google_vision", warning="google_cloud_vision not installed")

        started = time.monotonic()
        try:
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=image_bytes)
            response = client.web_detection(image=image)
            web = getattr(response, "web_detection", None)
            
            prov_id = log_request("google_vision", "POST", "https://vision.googleapis.com/v1/images:annotate",
                                  {"features": ["WEB_DETECTION"]}, 200, (time.monotonic() - started) * 1000, len(image_bytes))

            if web is None:
                LOGGER.warning("unexpected_shape engine=google_vision top_level_keys=[]")
                return SearchResponse(engine="google_vision", provenance_id=prov_id, warning="no_web_detection")

            candidates = []
            pages = getattr(web, "pages_with_matching_images", []) or []
            for page in pages:
                url = getattr(page, "url", None)
                if url:
                    candidates.append(Candidate(
                        url=url,
                        title=getattr(page, "page_title", None),
                        source_engine="google_vision",
                        match_confidence_hint="visual",
                        provenance_id=prov_id,
                    ))

            entities = [getattr(e, "description", "") for e in getattr(web, "web_entities", []) if getattr(e, "score", 0) > 0.6]
            labels = [getattr(l, "label", "") for l in getattr(web, "best_guess_labels", [])]
            query = " ".join([w for w in (entities + labels) if w.strip()][:4])

            return SearchResponse(
                engine="google_vision",
                candidates=candidates,
                provenance_id=prov_id,
                raw_payload={"query": query},
            )
        except Exception as err:
            LOGGER.warning("Google Vision search error: %s", err)
            return SearchResponse(engine="google_vision", warning=f"google_vision_failed: {err}")
