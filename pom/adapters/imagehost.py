"""Image host adapter to make query image publicly reachable (AH-7, §3)."""

from __future__ import annotations

import logging
from typing import Any

from pom.config import CONFIG

LOGGER = logging.getLogger(__name__)


def host_query_image(image_bytes: bytes) -> str | None:
    """Host query image on public IPFS gateway or configured image host."""
    from pom.adapters.ipfs_pinata import pin_bytes_to_pinata
    
    res = pin_bytes_to_pinata(image_bytes, "query_image.jpg")
    if res.get("pinned"):
        return res.get("gateway_url")
    return None
