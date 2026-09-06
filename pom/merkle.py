"""Evidence Merkle tree calculation over artifacts (§3)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence


def hash_leaf(data: bytes) -> bytes:
    """Hash a leaf node with 0x00 domain separator."""
    return hashlib.sha256(b"\x00" + data).digest()


def hash_internal(left: bytes, right: bytes) -> bytes:
    """Hash internal node with 0x01 domain separator and sorted children."""
    if left > right:
        left, right = right, left
    return hashlib.sha256(b"\x01" + left + right).digest()


def compute_merkle_root(leaf_bytes_list: Sequence[bytes]) -> str:
    """Compute the Merkle root of an ordered list of leaf bytes.
    
    Returns '0x' + hex string. Empty list returns 32 zeros.
    """
    if not leaf_bytes_list:
        return "0x" + "0" * 64

    current_layer = [hash_leaf(data) for data in leaf_bytes_list]
    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            if i + 1 < len(current_layer):
                right = current_layer[i + 1]
            else:
                right = left  # Duplicate last element if odd count
            next_layer.append(hash_internal(left, right))
        current_layer = next_layer

    return "0x" + current_layer[0].hex()


def compute_evidence_dir_merkle_root(evidence_dir: Path) -> tuple[str, dict[str, str]]:
    """Compute Merkle root over all files in the evidence directory.
    
    Returns (merkle_root_hex, dict_of_file_hashes).
    """
    if not evidence_dir.is_dir():
        return "0x" + "0" * 64, {}

    files = sorted([p for p in evidence_dir.iterdir() if p.is_file()])
    file_hashes = {}
    leaf_bytes = []
    
    for f in files:
        data = f.read_bytes()
        f_hash = "0x" + hashlib.sha256(data).hexdigest()
        file_hashes[f.name] = f_hash
        leaf_bytes.append(data)

    root = compute_merkle_root(leaf_bytes)
    return root, file_hashes
