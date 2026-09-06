"""Pinata IPFS adapter for encrypted bundle pinning (§3, §11).

Guarantees:
- §11: POST https://api.pinata.cloud/pinning/pinFileToIPFS
- Header: Authorization: Bearer <PINATA_JWT>
- Read IpfsHash from response (verified against probes/pinata.json).
- If pinning fails: WARN, set cid: None, and STILL ANCHOR (INV-7).
"""

from __future__ import annotations

import logging
import os
import time

from typing import Any

from common.http_utils import create_session, response_meta
from pom.config import CONFIG
from pom.provenance import log_request

LOGGER = logging.getLogger(__name__)


def pin_bytes_to_pinata(data_bytes: bytes, filename: str = "bundle.enc") -> dict[str, Any]:
    """Upload/Pin bytes to Pinata IPFS gateway (§11)."""
    jwt = os.environ.get("PINATA_JWT") if "PINATA_JWT" in os.environ else CONFIG.pinata_jwt
    
    if not jwt:
        LOGGER.warning("pinata_skipped: no PINATA_JWT configured")
        return {
            "pinned": False,
            "cid": None,
            "gateway_url": None,
            "warning": "pinata_skipped: no PINATA_JWT configured",
        }

    started = time.monotonic()
    try:
        session = create_session()
        files = {
            "file": (filename, data_bytes, "application/octet-stream")
        }
        headers = {"Authorization": f"Bearer {jwt}"}
        resp = session.post(
            "https://api.pinata.cloud/pinning/pinFileToIPFS",
            files=files,
            headers=headers,
            timeout=15,
        )
        status, size = response_meta(resp)
        log_request("pinata_pin", "POST", "https://api.pinata.cloud/pinning/pinFileToIPFS",
                    {"filename": filename}, status, (time.monotonic() - started) * 1000, size,
                    response_data=getattr(resp, "content", None))

        if status is not None and status == 200:
            res_data = resp.json() if hasattr(resp, "json") else {}
            if not isinstance(res_data, dict):
                LOGGER.warning("unexpected_shape engine=pinata top_level_keys=[]")
                return {
                    "pinned": False,
                    "cid": None,
                    "gateway_url": None,
                    "warning": "pinata_invalid_json_response",
                }

            pin_cid = res_data.get("IpfsHash")
            if not pin_cid:
                top_keys = sorted(list(res_data.keys()))
                LOGGER.warning("unexpected_shape engine=pinata top_level_keys=%s", top_keys)
                return {
                    "pinned": False,
                    "cid": None,
                    "gateway_url": None,
                    "warning": "pinata_missing_IpfsHash",
                }

            return {
                "pinned": True,
                "cid": pin_cid,
                "gateway_url": f"https://gateway.pinata.cloud/ipfs/{pin_cid}",
                "timestamp": res_data.get("Timestamp"),
            }
        else:
            LOGGER.warning("pinata_upload_failed: status %s", status)
            return {
                "pinned": False,
                "cid": None,
                "gateway_url": None,
                "warning": f"pinata_upload_failed_status_{status}",
            }
    except Exception as err:
        LOGGER.warning("Pinata upload exception: %s", err)
        return {
            "pinned": False,
            "cid": None,
            "gateway_url": None,
            "warning": f"pinata_upload_exception: {err}",
        }
