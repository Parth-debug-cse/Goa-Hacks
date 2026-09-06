"""Provenance: an append-only JSONL log of every external HTTP call a run makes.

Each call to :func:`log_request` records one JSON line under
``evidence/requests.jsonl`` (created on demand) and returns a monotonic
``p_0001``-style id that downstream records (e.g. ``CandidateURL.provenance_id``)
can reference. Secrets appearing under known key names are redacted before the
record is ever written.
"""

from __future__ import annotations

import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Repo-root evidence directory; overridable in tests via the module attribute.
LOG_PATH = Path(__file__).resolve().parent.parent / "evidence" / "requests.jsonl"

# Key names whose values are replaced with "[REDACTED]" (case-insensitive).
_REDACTED_KEYS = {
    "api_key",
    "key",
    "authorization",
    "x-api-key",
    "ocp-apim-subscription-key",
}

_counter = itertools.count(1)


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``params`` with any secret-valued key redacted."""
    return {
        str(key): "[REDACTED]" if str(key).lower() in _REDACTED_KEYS else value
        for key, value in (params or {}).items()
    }


def log_request(
    engine: str,
    method: str,
    url: str,
    params: dict[str, Any],
    status: int | None,
    latency_ms: float,
    response_bytes: int | None,
    error: str | None = None,
) -> str:
    """Append one JSONL provenance line and return its monotonic id.

    This is write-only and never raises: provenance failures must not take the
    pipeline down, so any filesystem problem is downgraded to a warning.
    """
    provenance_id = f"p_{next(_counter):04d}"
    record: dict[str, Any] = {
        "provenance_id": provenance_id,
        "engine": engine,
        "method": method,
        "url": url,
        "params": _redact_params(params or {}),
        "status": status,
        "latency_ms": round(float(latency_ms or 0.0), 1),
        "response_bytes": response_bytes,
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    except OSError as err:
        LOGGER.warning("provenance_log_unavailable: could not write %s: %s", LOG_PATH, err)
    return provenance_id