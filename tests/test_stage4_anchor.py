"""Tests for Stage 4 blockchain anchoring (INV-1, INV-4, INV-5, INV-7)."""

import json
from stage4_anchor import build_canonical_record, process_anchoring


def test_build_canonical_record():
    stage3_data = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/testuser",
        "matched_image_url": "https://media.test/photo.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.42,
            "adaface_cosine_similarity": 0.38,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 2,
    }
    record = build_canonical_record(stage3_data)
    assert record["schema_version"] == "1.0.0"
    assert record["match_found"] is True
    assert record["matched_page_url"] == "https://linkedin.com/in/testuser"
    assert record["consent_confirmed"] is True


def test_process_anchoring_refuses_no_match():
    # INV-1: Never anchor if match_found is False
    stage3_no_match = {
        "match_found": False,
        "candidates_tried": 5,
        "warnings": ["no_candidates_to_verify"],
    }
    result = process_anchoring(stage3_no_match)
    assert result["anchored"] is False
    assert result["reason"] == "no_match_found"
    assert "record_hash_sha256" not in result


def test_process_anchoring_generates_valid_receipt(tmp_path):
    stage3_match = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/alice",
        "matched_image_url": "https://media.alice/photo.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.45,
            "adaface_cosine_similarity": 0.39,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 1,
    }
    out_file = str(tmp_path / "receipt.json")
    receipt = process_anchoring(stage3_match, output_path=out_file)

    assert receipt["anchored"] is True
    assert receipt["record_hash_sha256"].startswith("0x")
    assert receipt["record_hash_keccak256"].startswith("0x")
    assert receipt["ipfs_cid"].startswith("b")
    assert receipt["tx_hash"].startswith("0x")
    assert receipt["record"]["matched_page_url"] == "https://linkedin.com/in/alice"

    # Verify receipt written to disk
    with open(out_file, "r", encoding="utf-8") as f:
        disk_receipt = json.load(f)
    assert disk_receipt["record_hash_sha256"] == receipt["record_hash_sha256"]
