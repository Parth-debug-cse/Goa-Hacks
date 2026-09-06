"""AES-256-GCM bundle encryption, subject commitment, and IPFS CID calculation (§3, §11).

Guarantees:
- INV-5: NO BIOMETRICS AND NO PII ON-CHAIN OR ON PUBLIC IPFS.
  bundle.enc is the AES-256-GCM encrypted tar of evidence/ with layout: nonce (12B) || ciphertext || tag (16B).
- Key = secrets.token_bytes(32), stored only in out/<run>/bundle.key (gitignored, never uploaded).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import secrets
import tarfile
from pathlib import Path
from typing import Any

from pom.canonical import canonical_json_bytes

LOGGER = logging.getLogger(__name__)


def compute_ipfs_cid_v1(data: bytes, codec: int = 0x55) -> str:
    """Compute deterministic IPFS CIDv1 base32 multihash for raw bytes."""
    digest = hashlib.sha256(data).digest()
    multihash = bytes([0x12, 0x20]) + digest
    cid_binary = bytes([0x01, codec]) + multihash
    raw_b32 = base64.b32encode(cid_binary).decode("ascii").rstrip("=").lower()
    return "b" + raw_b32


def compute_subject_commitment(face_embedding: list[float], salt_bytes: bytes) -> str:
    """Compute zero-knowledge subject commitment: SHA-256(canonical_embedding || salt)."""
    emb_bytes = json.dumps(face_embedding, separators=(",", ":")).encode("utf-8")
    return "0x" + hashlib.sha256(emb_bytes + salt_bytes).hexdigest()


def create_evidence_tar(evidence_dir: Path) -> bytes:
    """Create a deterministic tarball of all files in the evidence directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        if evidence_dir.is_dir():
            for p in sorted(evidence_dir.iterdir(), key=lambda x: x.name):
                if p.is_file():
                    ti = tar.gettarinfo(str(p), arcname=p.name)
                    ti.mtime = 0
                    ti.uid = 0
                    ti.gid = 0
                    ti.uname = ""
                    ti.gname = ""
                    with open(p, "rb") as f:
                        tar.addfile(ti, f)
    return buf.getvalue()


def encrypt_evidence_bundle(
    evidence: Path | bytes,
    key: bytes | None = None,
) -> tuple[bytes, bytes, str]:
    """Encrypt evidence tarball using AES-256-GCM (§11).
    
    Layout: nonce (12 bytes) || ciphertext || tag (16 bytes).
    Key: 32 random bytes (secrets.token_bytes(32)).
    
    Returns:
      (bundle_enc_bytes, key_bytes, bundle_sha256_hex)
    """
    if isinstance(evidence, (Path, str)):
        raw_tar_bytes = create_evidence_tar(Path(evidence))
    else:
        raw_tar_bytes = evidence

    if key is None:
        key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(12)

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        # AESGCM.encrypt returns ciphertext + 16-byte authentication tag
        ct_and_tag = aesgcm.encrypt(nonce, raw_tar_bytes, None)
    except ImportError:
        from Crypto.Cipher import AES
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(raw_tar_bytes)
        ct_and_tag = ciphertext + tag

    # Binary layout: nonce (12B) || ciphertext || tag (16B)
    bundle_enc_bytes = nonce + ct_and_tag
    bundle_sha256 = "0x" + hashlib.sha256(bundle_enc_bytes).hexdigest()

    return bundle_enc_bytes, key, bundle_sha256


def decrypt_evidence_bundle(bundle_enc_bytes: bytes, key: bytes) -> bytes:
    """Decrypt bundle.enc (nonce || ciphertext || tag) back to raw tar bytes."""
    if len(bundle_enc_bytes) < 28:
        raise ValueError("Invalid bundle.enc: payload smaller than nonce (12B) + tag (16B)")

    nonce = bundle_enc_bytes[:12]
    ct_and_tag = bundle_enc_bytes[12:]

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct_and_tag, None)
    except ImportError:
        from Crypto.Cipher import AES
        ciphertext = ct_and_tag[:-16]
        tag = ct_and_tag[-16:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag)


# Backward-compatible helpers
def encrypt_bundle_aes_gcm(plaintext: bytes, key: bytes | None = None) -> tuple[bytes, dict[str, str], bytes]:
    bundle_enc, k, sha = encrypt_evidence_bundle(plaintext, key)
    nonce = bundle_enc[:12]
    ct_and_tag = bundle_enc[12:]
    meta = {
        "algorithm": "AES-256-GCM",
        "key_hex": "0x" + k.hex(),
        "key_b64": base64.b64encode(k).decode("ascii"),
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "bundle_sha256": sha,
    }
    return bundle_enc, meta, k


def decrypt_bundle_aes_gcm(bundle_bytes: bytes, key_b64: str) -> bytes:
    k = base64.b64decode(key_b64)
    # Check if raw binary or json wrapped
    try:
        data = json.loads(bundle_bytes.decode("utf-8"))
        if isinstance(data, dict) and "nonce" in data and "ciphertext" in data:
            nonce = base64.b64decode(data["nonce"])
            ct = base64.b64decode(data["ciphertext"])
            tag = base64.b64decode(data.get("tag", ""))
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return AESGCM(k).decrypt(nonce, ct + tag, None)
    except Exception:
        pass
    return decrypt_evidence_bundle(bundle_bytes, k)
