"""Small, mockable HTTP helpers shared by the search and verification stages."""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 10
USER_AGENT = "face-chain-verify/1.0 (consent-confirmed demo)"


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def request(
    method: str,
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> requests.Response:
    """Make one request with the shared timeout and retry policy."""
    client = session or create_session()
    return client.request(method, url, timeout=timeout, **kwargs)


def response_meta(response: Any) -> tuple[int | None, int | None]:
    """Best-effort (status_code, body_bytes) from a response object.

    Tolerates mocks and streamed responses so provenance logging can stay
    defensive: a missing status or body simply becomes ``None`` instead of
    crashing the pipeline.
    """
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        status = None
    content = getattr(response, "content", None)
    size = len(content) if isinstance(content, (bytes, bytearray)) else None
    return status, size
