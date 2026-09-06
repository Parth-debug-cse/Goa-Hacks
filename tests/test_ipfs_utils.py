"""Tests for IPFS CID calculation and AES-256-GCM encryption (INV-5, INV-7)."""

import json
from common.ipfs_utils import (
    compute_ipfs_cid_v0,
    compute_ipfs_cid_v1,
    decrypt_payload_aes_gcm,
    encrypt_payload_aes_gcm,
    pin_to_pinata,
)


def test_ipfs_cid_v1_deterministic():
    data = b'{"hello":"world"}'
    cid1 = compute_ipfs_cid_v1(data)
    cid2 = compute_ipfs_cid_v1(data)
    assert cid1 == cid2
    assert cid1.startswith("b")  # Base32 CIDv1 prefix


def test_ipfs_cid_v0_format():
    data = b"test payload"
    cid0 = compute_ipfs_cid_v0(data)
    assert cid0.startswith("Qm")


def test_aes_gcm_encryption_roundtrip():
    plaintext = b'{"match":true,"score":0.42}'
    bundle_bytes, meta = encrypt_payload_aes_gcm(plaintext)
    
    # Bundle should be valid JSON containing ciphertext, nonce, tag
    bundle = json.loads(bundle_bytes.decode("utf-8"))
    assert "ciphertext" in bundle
    assert "nonce" in bundle
    assert "tag" in bundle
    # INV-5: Plaintext must not appear in the encrypted bundle
    assert b"match" not in bundle_bytes

    # Decrypt and verify exact match
    decrypted = decrypt_payload_aes_gcm(bundle_bytes, meta["key_b64"])
    assert decrypted == plaintext


from unittest.mock import patch


def test_pinata_graceful_degradation_without_keys():
    warnings = []
    with patch.dict("os.environ", {"PINATA_JWT": "", "PINATA_API_KEY": "", "PINATA_API_SECRET": ""}):
        result = pin_to_pinata(b"encrypted_data", "bafytestcid", warnings)
    assert result["pinned"] is False
    assert result["cid"] == "bafytestcid"
    assert any("pinata_upload_skipped" in w for w in warnings)

