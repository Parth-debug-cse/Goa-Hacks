"""Tests for canonical JSON serialization and byte-exact reproducibility (INV-4)."""

import hashlib
import json
from common.canonical_json import (
    canonical_json_bytes,
    canonical_json_dumps,
    compute_record_hashes,
    keccak256_hex,
    sha256_hex,
)


def test_canonical_json_formatting():
    obj = {
        "z_field": "last",
        "a_field": "first",
        "nested": {
            "b": 2,
            "a": 1,
        },
        "list": [3, 2, 1],
    }
    dumped = canonical_json_dumps(obj)
    # Check that keys are alphabetically sorted and without spaces after separators
    expected = '{"a_field":"first","list":[3,2,1],"nested":{"a":1,"b":2},"z_field":"last"}'
    assert dumped == expected
    assert canonical_json_bytes(obj) == expected.encode("utf-8")


def test_byte_exact_sha256_hash():
    obj = {"test": 123, "name": "alice"}
    canonical = '{"name":"alice","test":123}'.encode("utf-8")
    expected_sha256 = "0x" + hashlib.sha256(canonical).hexdigest()
    
    assert sha256_hex(canonical) == expected_sha256
    assert compute_record_hashes(obj)["sha256"] == expected_sha256


def test_keccak256_hex():
    data = b"hello world"
    keccak = keccak256_hex(data)
    assert keccak.startswith("0x")
    assert len(keccak) == 66  # "0x" + 64 hex chars
