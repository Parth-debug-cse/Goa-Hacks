"""PeopleDataLabs adapter with defensive shape parsing (AH-2, AH-6)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from common.http_utils import create_session, response_meta
from common.provenance import log_request

LOGGER = logging.getLogger(__name__)


def enrich_person_profile(url: str, api_key: str) -> tuple[dict[str, Any], list[str]]:
    """Enrich a LinkedIn/X profile using PeopleDataLabs API."""
    warnings: list[str] = []
    params = {"profile": url, "min_likelihood": 4}
    started = time.monotonic()
    
    try:
        session = create_session()
        response = session.get(
            "https://api.peopledatalabs.com/v5/person/enrich",
            headers={"X-Api-Key": api_key},
            params=params,
            timeout=10,
        )
        status, size = response_meta(response)
        log_request("pdl", "GET", "https://api.peopledatalabs.com/v5/person/enrich",
                    {**params, "X-Api-Key": api_key},
                    status, (time.monotonic() - started) * 1000, size)

        if status is not None and status == 404:
            return {"attempted": True, "matched": False}, warnings
        if status is not None and status >= 400:
            warnings.append(f"pdl_http_error: {status}")
            return {"attempted": True, "matched": False}, warnings

        payload = response.json() if hasattr(response, "json") else {}
        if not isinstance(payload, dict):
            LOGGER.warning("unexpected_shape engine=pdl top_level_keys=[]")
            warnings.append("pdl_invalid_response_shape")
            return {"attempted": True, "matched": False}, warnings

        data = payload.get("data")
        if data is None:
            top_keys = sorted(list(payload.keys()))
            LOGGER.warning("unexpected_shape engine=pdl top_level_keys=%s", top_keys)
            return {"attempted": True, "matched": False}, warnings

        if not isinstance(data, dict):
            return {"attempted": True, "matched": False}, warnings

        result: dict[str, Any] = {
            "attempted": True,
            "matched": bool(data),
            "likelihood": payload.get("likelihood"),
            "full_name": data.get("full_name"),
            "linkedin_url": data.get("linkedin_url"),
            "job_title": data.get("job_title"),
            "job_company_name": data.get("job_company_name"),
            "location_name": data.get("location_name"),
            "raw_pdl_data": data,
        }
        return result, warnings
    except Exception as err:
        warnings.append(f"pdl_failed: {err}")
        return {"attempted": True, "matched": False}, warnings
