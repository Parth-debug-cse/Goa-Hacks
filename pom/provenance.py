"""Provenance logger recording external HTTP requests and response hashes (§3)."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pom.config import CONFIG

LOGGER = logging.getLogger(__name__)

# Default evidence log path, overridable by run_id
_DEFAULT_LOG_PATH = CONFIG.root_dir / "evidence" / "requests.jsonl"
_active_log_path = _DEFAULT_LOG_PATH
_counter = itertools.count(1)

_REDACTED_KEYS = {
    "api_key",
    "key",
    "authorization",
    "x-api-key",
    "ocp-apim-subscription-key",
    "pinata_api_key",
    "pinata_secret_api_key",
    "pom_private_key",
    "private_key",
}


def set_provenance_log_path(path: Path) -> None:
    """Set the active requests.jsonl path for the current run."""
    global _active_log_path
    _active_log_path = path


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Redact secret parameters: log key PRESENCE ('[PRESENT]'), never key value."""
    result = {}
    for key, value in (params or {}).items():
        k_lower = str(key).lower()
        if k_lower in _REDACTED_KEYS or "key" in k_lower or "secret" in k_lower or "token" in k_lower or "jwt" in k_lower:
            result[str(key)] = "[PRESENT]"
        else:
            result[str(key)] = value
    return result


def compute_bytes_sha256(data: bytes | None) -> str | None:
    """Compute SHA-256 hash of response bytes."""
    if data is None:
        return None
    return "0x" + hashlib.sha256(data).hexdigest()


def log_request(
    engine: str,
    method: str,
    url: str,
    params: dict[str, Any],
    status: int | None,
    latency_ms: float,
    response_bytes: int | None,
    response_data: bytes | None = None,
    error: str | None = None,
) -> str:
    """Append one JSONL provenance line and return its monotonic p_0001 id."""
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
        "response_sha256": compute_bytes_sha256(response_data),
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _active_log_path.parent.mkdir(parents=True, exist_ok=True)
        with _active_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    except OSError as err:
        LOGGER.warning("provenance_log_unavailable: could not write %s: %s", _active_log_path, err)
    return provenance_id
