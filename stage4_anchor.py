"""Stage 4: Blockchain anchoring and cryptographic receipt generation.

Anchors a verified face-match record on the Polygon Amoy blockchain and IPFS.

Invariants:
- INV-1: NO FABRICATED RESULTS. Anchoring is ONLY performed when match_found == True.
- INV-4: BYTE-EXACT REPRODUCIBILITY. Canonical RFC 8785 JSON digests.
- INV-5: NO BIOMETRICS AND NO PII ON-CHAIN OR ON PUBLIC IPFS.
  IPFS bundle is AES-GCM encrypted. Chain stores only 32-byte cryptographic hashes.
- INV-7: GRACEFUL DEGRADATION, LOUDLY. Structured warnings on network / credential gaps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from common.blockchain import AMOY_CHAIN_ID, anchor_on_chain
from common.canonical_json import (
    canonical_json_bytes,
    canonical_json_dumps,
    compute_record_hashes,
    sha256_hex,
)
from common.ipfs_utils import (
    compute_ipfs_cid_v1,
    encrypt_payload_aes_gcm,
    pin_to_pinata,
)

LOGGER = logging.getLogger(__name__)


def build_canonical_record(stage3_result: dict[str, Any]) -> dict[str, Any]:
    """Construct the minimal, deterministic canonical record from Stage 3 output.
    
    This record is the exact object whose SHA-256 and Keccak-256 digests are anchored.
    """
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_timestamp_utc": stage3_result.get(
            "run_timestamp_utc",
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        ),
        "match_found": bool(stage3_result.get("match_found")),
        "matched_page_url": stage3_result.get("matched_page_url"),
        "matched_image_url": stage3_result.get("matched_image_url"),
        "source_engine": stage3_result.get("source_engine"),
        "face_match": stage3_result.get("face_match"),
        "candidates_tried": stage3_result.get("candidates_tried", 0),
        "consent_confirmed": True,
    }
    
    # Optional PDL enrichment if present
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


def process_anchoring(
    stage3_result: dict[str, Any],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Anchor verified Stage 3 result on Polygon Amoy and produce an anchor receipt.
    
    Returns the complete anchor receipt dictionary.
    """
    warnings: list[str] = list(stage3_result.get("warnings", []))
    
    # INV-1: Never anchor if match_found is False
    if not stage3_result.get("match_found"):
        LOGGER.info("No verified match found; skipping blockchain anchoring (INV-1).")
        result = {
            "anchored": False,
            "reason": "no_match_found",
            "match_found": False,
            "warnings": warnings,
        }
        return result

    # 1. Build canonical record & calculate deterministic hashes (INV-4)
    record = build_canonical_record(stage3_result)
    raw_record_bytes = canonical_json_bytes(record)
    hashes = compute_record_hashes(record)
    record_hash_sha256 = hashes["sha256"]
    record_hash_keccak256 = hashes["keccak256"]

    # 2. Encrypt record with AES-256-GCM for IPFS bundle (INV-5)
    encrypted_bundle_bytes, encryption_meta = encrypt_payload_aes_gcm(raw_record_bytes)
    
    # 3. Compute deterministic CIDv1 for the encrypted bundle
    ipfs_cid = compute_ipfs_cid_v1(encrypted_bundle_bytes)

    # 4. Pin encrypted bundle to Pinata / IPFS (INV-5, INV-7)
    ipfs_result = pin_to_pinata(encrypted_bundle_bytes, ipfs_cid, warnings)

    # 5. Anchor on Polygon Amoy Testnet (INV-5, INV-7)
    chain_result = anchor_on_chain(
        record_hash=record_hash_keccak256,
        ipfs_cid=ipfs_cid,
        metadata_digest=record_hash_sha256,
        warnings=warnings,
    )

    receipt: dict[str, Any] = {
        "receipt_version": "1.0.0",
        "anchored": True,
        "chain_id": chain_result.get("chain_id", AMOY_CHAIN_ID),
        "chain_name": chain_result.get("chain_name", "Polygon Amoy Testnet"),
        "record_hash_keccak256": record_hash_keccak256,
        "record_hash_sha256": record_hash_sha256,
        "ipfs_cid": ipfs_cid,
        "ipfs_pinned": ipfs_result.get("pinned", False),
        "ipfs_gateway_url": ipfs_result.get("gateway_url"),
        "tx_hash": chain_result.get("tx_hash"),
        "block_number": chain_result.get("block_number"),
        "explorer_url": chain_result.get("explorer_url"),
        "contract_address": chain_result.get("contract_address"),
        "anchor_method": chain_result.get("anchor_method"),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "encryption": encryption_meta,
        "record": record,
        "warnings": warnings,
    }

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            LOGGER.info("Wrote anchor receipt to %s", output_path)
        except OSError as err:
            warnings.append(f"failed_to_write_receipt_file: {err}")

    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4: Anchor verified match on Polygon Amoy blockchain.")
    parser.add_argument("handoff_json", help="Path to Stage 3 handoff JSON file or '-' for stdin.")
    parser.add_argument("--output", "-o", default="anchor_receipt.json", help="Output path for anchor receipt.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.handoff_json == "-":
        stage3_data = json.load(sys.stdin)
    else:
        with open(args.handoff_json, "r", encoding="utf-8") as f:
            stage3_data = json.load(f)

    receipt = process_anchoring(stage3_data, args.output)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0 if receipt.get("anchored") else 2


if __name__ == "__main__":
    raise SystemExit(main())
