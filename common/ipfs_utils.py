"""IPFS and decentralized content-addressing utilities with AES-GCM encryption.

Adheres strictly to:
- INV-5: NO BIOMETRICS AND NO PII ON-CHAIN OR ON PUBLIC IPFS.
  All IPFS bundles containing match records/embeddings are AES-256-GCM encrypted.
  The public IPFS and blockchain store only ciphertext/commitments.
- INV-7: GRACEFUL DEGRADATION, LOUDLY.
  If remote Pinata pinning fails or keys are missing, logs structured warning
  and computes deterministic standard CIDv1 multihashes locally.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from typing import Any

from common.canonical_json import canonical_json_bytes
from common.http_utils import create_session, response_meta
from common.provenance import log_request

LOGGER = logging.getLogger(__name__)

# Base58 and Base32 character sets for standard IPFS CIDs
_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE32_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"


def _b58encode(data: bytes) -> str:
    """Encode bytes into standard Bitcoin Base58 string."""
    n = int.from_bytes(data, "big")
    chars = []
    while n > 0:
        n, r = divmod(n, 58)
        chars.append(_BASE58_ALPHABET[r])
    # Leading zeros
    pad = 0
    for byte in data:
        if byte == 0:
            pad += 1
        else:
            break
    return (_BASE58_ALPHABET[0] * pad) + "".join(reversed(chars))


def _b32encode_no_pad(data: bytes) -> str:
    """Encode bytes into RFC 4648 Base32 without padding, lower-cased."""
    raw = base64.b32encode(data).decode("ascii").rstrip("=").lower()
    return raw


def compute_ipfs_cid_v1(data: bytes, codec: int = 0x55) -> str:
    """Compute standard IPFS CIDv1 (base32, raw/dag-pb, sha2-256 multihash).
    
    Format: 'b' + base32( 0x01 (cidv1) + codec (0x55 = raw) + 0x12 (sha2-256) + 0x20 (32 bytes) + sha256_digest )
    """
    digest = hashlib.sha256(data).digest()
    # CIDv1 binary header: 0x01 (version 1), codec, 0x12 (sha2-256), 0x20 (len 32)
    multihash = bytes([0x12, 0x20]) + digest
    cid_binary = bytes([0x01, codec]) + multihash
    return "b" + _b32encode_no_pad(cid_binary)


def compute_ipfs_cid_v0(data: bytes) -> str:
    """Compute standard legacy IPFS CIDv0 (Base58btc 'Qm...')."""
    digest = hashlib.sha256(data).digest()
    multihash = bytes([0x12, 0x20]) + digest
    return _b58encode(multihash)


def encrypt_payload_aes_gcm(plaintext: bytes, key: bytes | None = None) -> tuple[bytes, dict[str, str]]:
    """Encrypt payload using AES-256-GCM.
    
    Returns (encrypted_bundle_bytes, metadata_dict).
    Metadata contains:
      - algorithm: 'AES-256-GCM'
      - key_b64: base64 encoded 256-bit symmetric key
      - nonce_b64: base64 encoded 12-byte IV/nonce
      - tag_b64: base64 encoded 16-byte authentication tag
    """
    if key is None:
        key = os.urandom(32)
    nonce = os.urandom(12)

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        # AESGCM.encrypt appends 16-byte tag to the ciphertext
        ct_and_tag = aesgcm.encrypt(nonce, plaintext, None)
        ciphertext = ct_and_tag[:-16]
        tag = ct_and_tag[-16:]
    except ImportError:
        # Fallback to PyCryptodome if cryptography is not installed
        from Crypto.Cipher import AES
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    encrypted_bundle = {
        "version": "1.0",
        "encryption": "AES-256-GCM",
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "tag": base64.b64encode(tag).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }
    bundle_bytes = json.dumps(encrypted_bundle, sort_keys=True).encode("utf-8")

    meta = {
        "algorithm": "AES-256-GCM",
        "key_b64": base64.b64encode(key).decode("ascii"),
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "tag_b64": base64.b64encode(tag).decode("ascii"),
    }
    return bundle_bytes, meta


def decrypt_payload_aes_gcm(encrypted_bundle_bytes: bytes, key_b64: str) -> bytes:
    """Decrypt an AES-256-GCM encrypted bundle back to original canonical bytes."""
    bundle = json.loads(encrypted_bundle_bytes.decode("utf-8"))
    key = base64.b64decode(key_b64)
    nonce = base64.b64decode(bundle["nonce"])
    tag = base64.b64decode(bundle["tag"])
    ciphertext = base64.b64decode(bundle["ciphertext"])

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext + tag, None)
        return plaintext
    except ImportError:
        from Crypto.Cipher import AES
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext


def pin_to_pinata(
    encrypted_bytes: bytes,
    cid: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Upload/Pin encrypted bundle to Pinata IPFS (if credentials configured).
    
    Adheres strictly to INV-5 (only encrypted bundle is uploaded) and INV-7 (graceful degradation).
    """
    jwt = os.environ.get("PINATA_JWT")
    api_key = os.environ.get("PINATA_API_KEY")
    api_secret = os.environ.get("PINATA_API_SECRET")

    if not jwt and not (api_key and api_secret):
        warnings.append("pinata_upload_skipped: no pinata credentials configured")
        return {"pinned": False, "cid": cid, "gateway_url": f"https://gateway.pinata.cloud/ipfs/{cid}"}

    headers: dict[str, str] = {}
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"
    else:
        headers["pinata_api_key"] = api_key or ""
        headers["pinata_secret_api_key"] = api_secret or ""

    started = time.monotonic()
    try:
        session = create_session()
        files = {
            "file": ("encrypted_proof.bin", encrypted_bytes, "application/octet-stream")
        }
        response = session.post(
            "https://api.pinata.cloud/pinning/pinFileToIPFS",
            files=files,
            headers=headers,
            timeout=15,
        )
        status, size = response_meta(response)
        log_request("pinata_pin", "POST", "https://api.pinata.cloud/pinning/pinFileToIPFS",
                    {"cid": cid}, status, (time.monotonic() - started) * 1000, size)

        if status is not None and status == 200:
            res_data = response.json()
            pinata_cid = res_data.get("IpfsHash", cid)
            return {
                "pinned": True,
                "cid": pinata_cid,
                "gateway_url": f"https://gateway.pinata.cloud/ipfs/{pinata_cid}",
                "timestamp": res_data.get("Timestamp"),
            }
        else:
            warnings.append(f"pinata_upload_failed: status {status}")
            return {"pinned": False, "cid": cid, "gateway_url": f"https://gateway.pinata.cloud/ipfs/{cid}"}
    except Exception as err:
        warnings.append(f"pinata_upload_exception: {err}")
        return {"pinned": False, "cid": cid, "gateway_url": f"https://gateway.pinata.cloud/ipfs/{cid}"}
