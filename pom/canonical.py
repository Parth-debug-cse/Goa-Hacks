"""Canonical JSON serialization, record_hash calculation, and dotted-path utilities (§3)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_dumps(obj: Any) -> str:
    """Serialize object to RFC 8785 deterministic canonical JSON string."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize object to RFC 8785 canonical UTF-8 bytes."""
    return canonical_json_dumps(obj).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    """Compute SHA-256 digest prefixed with '0x'."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "0x" + hashlib.sha256(data).hexdigest()


def keccak256_hex(data: bytes | str) -> str:
    """Compute Keccak-256 digest prefixed with '0x'."""
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


def compute_record_hash(record: dict[str, Any]) -> str:
    """Calculate the canonical record_hash (SHA-256) for a record dict (INV-4)."""
    return sha256_hex(canonical_json_bytes(record))


PATH_ALIASES: dict[str, str] = {
    "match.similarity.score": "face_match.arcface_cosine_similarity",
    "match.similarity": "face_match.arcface_cosine_similarity",
    "similarity": "face_match.arcface_cosine_similarity",
    "url": "matched_page_url",
    "page_url": "matched_page_url",
    "timestamp": "run_timestamp_utc",
}


def resolve_dotted_path(data: dict[str, Any], path: str) -> str:
    """Resolve aliases if the target key exists under the canonical name."""
    if path in PATH_ALIASES:
        canonical_target = PATH_ALIASES[path]
        tokens = canonical_target.split(".")
        curr = data
        found = True
        for token in tokens:
            if isinstance(curr, dict) and token in curr:
                curr = curr[token]
            else:
                found = False
                break
        if found:
            return canonical_target
    return path


def dotted_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Retrieve value at dotted path (e.g. 'face_match.arcface_cosine_similarity')."""
    resolved = resolve_dotted_path(data, path)
    tokens = resolved.split(".")
    curr = data
    for token in tokens:
        if isinstance(curr, dict) and token in curr:
            curr = curr[token]
        else:
            return default
    return curr


def dotted_set(data: dict[str, Any], path: str, value: Any) -> None:
    """Set value at dotted path in-place."""
    resolved = resolve_dotted_path(data, path)
    tokens = resolved.split(".")
    curr = data
    for token in tokens[:-1]:
        if token not in curr or not isinstance(curr[token], dict):
            curr[token] = {}
        curr = curr[token]
    curr[tokens[-1]] = value

