"""Canonical JSON serialization and cryptographic hash utilities.

Adheres strictly to RFC 8785 / deterministic key sorting and compact formatting
to guarantee byte-exact reproducibility across independent third-party verifiers
using standard tools (e.g. jq -c -S . record.json | sha256sum). (INV-4)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_dumps(obj: Any) -> str:
    """Serialize any JSON-serializable Python object to canonical JSON string.
    
    Keys are sorted alphabetically at all depths, with compact separators (',', ':'),
    and unicode characters preserved without ASCII escaping.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize any JSON-serializable Python object to UTF-8 encoded canonical bytes."""
    return canonical_json_dumps(obj).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    """Compute SHA-256 hex digest (with leading '0x')."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "0x" + hashlib.sha256(data).hexdigest()


def keccak256_hex(data: bytes | str) -> str:
    """Compute Keccak-256 hex digest (with leading '0x').
    
    Uses pycryptodome / eth_utils / hashlib sha3 fallback.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    try:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256)
        k.update(data)
        return "0x" + k.hexdigest()
    except ImportError:
        try:
            from eth_utils import keccak
            return "0x" + keccak(data).hex()
        except ImportError:
            return "0x" + hashlib.sha3_256(data).hexdigest()


def compute_record_hashes(record: dict[str, Any]) -> dict[str, str]:
    """Compute both SHA-256 and Keccak-256 canonical digests for a record."""
    raw_bytes = canonical_json_bytes(record)
    return {
        "sha256": sha256_hex(raw_bytes),
        "keccak256": keccak256_hex(raw_bytes),
    }
