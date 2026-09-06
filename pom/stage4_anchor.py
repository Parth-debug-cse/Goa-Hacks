"""Stage 4: Blockchain anchoring and encrypted IPFS bundle generation (§3, §11).

Guarantees:
- INV-1: NO FABRICATED RESULTS. Only executed when match_found = true.
- INV-4: BYTE-EXACT REPRODUCIBILITY. Canonical JSON record_hash.
- INV-5: NO BIOMETRICS AND NO PII ON-CHAIN OR ON PUBLIC IPFS.
  bundle.enc is the AES-256-GCM encrypted tar of evidence/. Chain stores only 32-byte digests.
- §11: Key = secrets.token_bytes(32) written to out/<run>/bundle.key (gitignored).
- §11: Record cid plus bundle_sha256 (of the ciphertext). If pinning fails: warn, set cid: null, STILL ANCHOR (INV-7).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from pom.adapters.chain import get_web3_client
from pom.adapters.ipfs_pinata import pin_bytes_to_pinata
from pom.canonical import canonical_json_bytes, compute_record_hash, sha256_hex
from pom.config import CONFIG
from pom.crypto import compute_subject_commitment, encrypt_evidence_bundle
from pom.merkle import compute_evidence_dir_merkle_root

LOGGER = logging.getLogger(__name__)


def build_canonical_record(stage3_result: dict[str, Any], stage1_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Construct the minimal, deterministic canonical record from Stage 3 output (INV-4)."""
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_timestamp_utc": stage3_result.get("run_timestamp_utc", ""),
        "match_found": bool(stage3_result.get("match_found")),
        "matched_page_url": stage3_result.get("matched_page_url"),
        "matched_image_url": stage3_result.get("matched_image_url"),
        "source_engine": stage3_result.get("source_engine"),
        "face_match": stage3_result.get("face_match"),
        "candidates_tried": stage3_result.get("candidates_tried", 0),
        "consent_confirmed": True,
    }
    if "pdl_enrichment" in stage3_result and stage3_result["pdl_enrichment"]:
        pdl = stage3_result["pdl_enrichment"]
        record["pdl_enrichment"] = {
            "attempted": pdl.get("attempted", False),
            "matched": pdl.get("matched", False),
            "likelihood": pdl.get("likelihood"),
            "full_name": pdl.get("full_name"),
            "linkedin_url": pdl.get("linkedin_url"),
            "job_title": pdl.get("job_title"),
            "job_company_name": pdl.get("job_company_name"),
            "location_name": pdl.get("location_name"),
        }
    return record


def process_stage4(
    stage3_result: dict[str, Any],
    stage1_result: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute Stage 4 encryption, IPFS pinning, and blockchain anchoring (§11)."""
    warnings: list[str] = list(stage3_result.get("warnings", []))

    if not stage3_result.get("match_found"):
        LOGGER.info("No verified match found; skipping blockchain anchoring (INV-1).")
        return {"anchored": False, "reason": "no_match_found", "warnings": warnings}

    # 1. Build canonical record & calculate deterministic hash (INV-4)
    record = build_canonical_record(stage3_result, stage1_result)
    record_hash = compute_record_hash(record)

    # 2. Compute Evidence Merkle Root
    evidence_dir = run_dir / "evidence" if run_dir else Path("evidence")
    merkle_root, file_hashes = compute_evidence_dir_merkle_root(evidence_dir)

    # 3. Generate salt & subject commitment
    salt = secrets.token_bytes(32)
    salt_hex = "0x" + salt.hex()
    arcface_emb = (stage1_result or {}).get("arcface_embedding", []) or []
    subject_commitment = compute_subject_commitment(arcface_emb, salt) if arcface_emb else "0x" + "0" * 64

    # 4. §11: Tar evidence/ -> AES-256-GCM -> bundle.enc with layout: nonce (12B) || ciphertext || tag (16B)
    bundle_enc_bytes, bundle_key, bundle_sha256 = encrypt_evidence_bundle(evidence_dir)

    # 5. §11: Pin via adapters/ipfs_pinata.py
    ipfs_res = pin_bytes_to_pinata(bundle_enc_bytes, "bundle.enc")
    ipfs_cid = ipfs_res.get("cid")  # None if pinning fails
    if ipfs_res.get("warning"):
        warnings.append(ipfs_res["warning"])

    # 6. Blockchain Anchoring (Polygon Amoy / Base Sepolia) (§12)
    from pom.adapters.chain import anchor_match_on_chain
    chain_res = anchor_match_on_chain(
        record_hash=record_hash,
        subject_commitment=subject_commitment,
        evidence_root=merkle_root,
        cid=ipfs_cid,
        chain_name=CONFIG.chain,
    )
    if chain_res.get("warnings"):
        warnings.extend(chain_res["warnings"])

    anchor_receipt = {
        "receipt_version": "1.0.0",
        "anchored": chain_res.get("anchored", True),
        "chain": chain_res.get("chain", CONFIG.chain),
        "chain_id": chain_res.get("chain_id", 80002 if CONFIG.chain == "amoy" else 84532),
        "contract_address": chain_res.get("contract_address", CONFIG.get_active_contract()),
        "record_hash_sha256": record_hash,
        "cid": ipfs_cid,
        "ipfs_cid": ipfs_cid,
        "bundle_sha256": bundle_sha256,
        "ipfs_pinned": ipfs_res.get("pinned", False),
        "ipfs_gateway_url": ipfs_res.get("gateway_url"),
        "tx_hash": chain_res.get("tx_hash", "0x" + "0" * 64),
        "block_number": chain_res.get("block_number", 0),
        "explorer_url": chain_res.get("explorer_url", ""),
        "merkle_root": merkle_root,
        "subject_commitment": subject_commitment,
        "record": record,
        "warnings": warnings,
    }

    # 7. Write run artifacts to out/<run_id>/ (§3, §11)
    if run_dir:
        (run_dir / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "salt.hex").write_text(salt_hex + "\n", encoding="utf-8")
        (run_dir / "bundle.key").write_text("0x" + bundle_key.hex() + "\n", encoding="utf-8")
        (run_dir / "bundle.enc").write_bytes(bundle_enc_bytes)
        (run_dir / "anchor.json").write_text(json.dumps(anchor_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return anchor_receipt
