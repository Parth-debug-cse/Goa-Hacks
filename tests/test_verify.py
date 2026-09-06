"""Tests for Independent Verification CLI and Tamper Detection (INV-4, INV-5)."""

import json
from stage4_anchor import process_anchoring
from verify import mutate_record_for_demo, verify_receipt


def _make_valid_receipt() -> dict:
    stage3_match = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/verifiedperson",
        "matched_image_url": "https://media.verified/photo.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.45,
            "adaface_cosine_similarity": 0.40,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 1,
    }
    return process_anchoring(stage3_match)


def test_verify_receipt_passes_on_valid_receipt():
    receipt = _make_valid_receipt()
    passed, errors = verify_receipt(receipt, quiet=True)
    assert passed is True
    assert len(errors) == 0


def test_verify_receipt_fails_when_similarity_mutated():
    receipt = _make_valid_receipt()
    # Deliberately modify similarity score by 0.001
    receipt["record"]["face_match"]["arcface_cosine_similarity"] = 0.451
    
    passed, errors = verify_receipt(receipt, quiet=True)
    assert passed is False
    assert any("TAMPER DETECTED" in err for err in errors)


def test_verify_receipt_fails_when_url_mutated():
    receipt = _make_valid_receipt()
    # Deliberately modify 1 character in URL
    receipt["record"]["matched_page_url"] = "https://linkedin.com/in/verifiedpersom"
    
    passed, errors = verify_receipt(receipt, quiet=True)
    assert passed is False
    assert any("TAMPER DETECTED" in err for err in errors)


def test_tamper_demo_mode_triggers_fail():
    receipt = _make_valid_receipt()
    passed, errors = verify_receipt(receipt, tamper_demo=True, quiet=True)
    assert passed is False
    assert any("TAMPER DETECTED" in err for err in errors)


def test_mutate_record_for_demo_fields():
    receipt = _make_valid_receipt()
    
    # Test similarity mutation
    tampered_sim, diff_sim = mutate_record_for_demo(receipt["record"], "similarity")
    assert tampered_sim["face_match"]["arcface_cosine_similarity"] != receipt["record"]["face_match"]["arcface_cosine_similarity"]
    assert "face_match" in diff_sim

    # Test URL mutation
    tampered_url, diff_url = mutate_record_for_demo(receipt["record"], "url")
    assert tampered_url["matched_page_url"] != receipt["record"]["matched_page_url"]
    assert "matched_page_url" in diff_url
