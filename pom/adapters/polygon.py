"""Polygon Amoy Web3 adapter with robust POA middleware compatibility (AH-7).

Ensures compatibility across web3.py v6 and v7:
- v6: geth_poa_middleware
- v7: ExtraDataToPOAMiddleware
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def inject_poa_middleware(w3: Any) -> None:
    """Inject POA middleware to prevent extraData length errors on Polygon Amoy & Base (AH-7)."""
    # First attempt web3.py v6 geth_poa_middleware (pinned standard)
    try:
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        return
    except (ImportError, AttributeError):
        pass

    # Then attempt web3.py v7 ExtraDataToPOAMiddleware
    try:
        from web3.middleware import ExtraDataToPOAMiddleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return
    except (ImportError, AttributeError):
        pass

    LOGGER.warning("Could not inject POA middleware on Web3 instance.")
