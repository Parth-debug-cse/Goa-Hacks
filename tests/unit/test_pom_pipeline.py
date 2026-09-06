"""Unit tests for pom pipeline stages and correctness invariants (INV-1 through INV-7, §11)."""

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from pom.canonical import canonical_json_bytes, compute_record_hash, dotted_get, dotted_set
from pom.crypto import (
    compute_ipfs_cid_v1,
    create_evidence_tar,
    decrypt_evidence_bundle,
    encrypt_evidence_bundle,
)
from pom.merkle import compute_merkle_root
from pom.stage0_manifest import setup_run_manifest
from pom.stage4_anchor import process_stage4
from pom.verifier import inject_tamper, verify_record_and_anchor


def test_canonical_byte_exact_reproducibility():
    record = {
        "z_field": "last",
        "a_field": "first",
        "match_found": True,
        "face_match": {"arcface_cosine_similarity": 0.45},
    }
    bytes1 = canonical_json_bytes(record)
    bytes2 = canonical_json_bytes(record)
    assert bytes1 == bytes2
    assert compute_record_hash(record) == compute_record_hash(record)


def test_merkle_root_empty_and_leaves():
    assert compute_merkle_root([]) == "0x" + "0" * 64
    leaves = [b"leaf1", b"leaf2", b"leaf3"]
    root = compute_merkle_root(leaves)
    assert root.startswith("0x")
    assert len(root) == 66


def test_milestone5_encrypted_evidence_tar_and_key(tmp_path):
    # Setup test evidence dir
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "requests.jsonl").write_text('{"id":"p_0001"}\n')
    (evidence_dir / "page.html").write_text("<html>Alice</html>")
    (evidence_dir / "input.jpg").write_bytes(b"\xff\xd8\xff\xe0testimage")

    # Encrypt
    bundle_enc_bytes, key, bundle_sha256 = encrypt_evidence_bundle(evidence_dir)
    assert len(key) == 32
    assert bundle_sha256.startswith("0x")
    assert len(bundle_sha256) == 66

    # Layout verification: nonce (12B) || ciphertext || tag (16B)
    assert len(bundle_enc_bytes) >= 28

    # Decrypt and verify tar
    decrypted_tar_bytes = decrypt_evidence_bundle(bundle_enc_bytes, key)
    with tarfile.open(fileobj=io.BytesIO(decrypted_tar_bytes)) as tar:
        names = tar.getnames()
        assert "requests.jsonl" in names
        assert "page.html" in names
        assert "input.jpg" in names


def test_milestone5_pinata_failure_sets_cid_null_and_still_anchors(tmp_path):
    stage3_data = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/authentic",
        "matched_image_url": "https://media.authentic/img.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.44,
            "adaface_cosine_similarity": 0.38,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 1,
    }
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "input.jpg").write_bytes(b"image")

    # When no PINATA_JWT is present, pinning fails gracefully (INV-7)
    with patch.dict("os.environ", {"PINATA_JWT": ""}):
        receipt = process_stage4(stage3_data, run_dir=tmp_path)

    assert receipt["anchored"] is True
    assert receipt["cid"] is None
    assert receipt["bundle_sha256"].startswith("0x")
    assert (tmp_path / "bundle.key").exists()
    assert (tmp_path / "bundle.enc").exists()
    assert any("pinata_skipped" in w for w in receipt["warnings"])


def test_dotted_get_set():
    d = {"face_match": {"arcface_cosine_similarity": 0.41}}
    assert dotted_get(d, "face_match.arcface_cosine_similarity") == 0.41
    dotted_set(d, "face_match.arcface_cosine_similarity", 0.42)
    assert dotted_get(d, "face_match.arcface_cosine_similarity") == 0.42


def test_verifier_passes_on_authentic_record(tmp_path):
    stage3_data = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/authentic",
        "matched_image_url": "https://media.authentic/img.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.44,
            "adaface_cosine_similarity": 0.38,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 1,
    }
    receipt = process_stage4(stage3_data, run_dir=tmp_path)
    passed, errors = verify_record_and_anchor(receipt, quiet=True)
    assert passed is True
    assert len(errors) == 0


def test_verifier_fails_instantly_on_tamper(tmp_path):
    stage3_data = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/authentic",
        "matched_image_url": "https://media.authentic/img.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.44,
            "adaface_cosine_similarity": 0.38,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 1,
    }
    receipt = process_stage4(stage3_data, run_dir=tmp_path)
    # Inject tamper
    passed, errors = verify_record_and_anchor(receipt, tamper_demo=True, quiet=True)
    assert passed is False
    assert any("TAMPER DETECTED" in err for err in errors)


def test_verifier_checks_bundle_bytes_and_detects_mismatch(tmp_path):
    stage3_data = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/authentic",
        "matched_image_url": "https://media.authentic/img.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.44,
            "adaface_cosine_similarity": 0.38,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 1,
    }
    receipt = process_stage4(stage3_data, run_dir=tmp_path)
    bundle_bytes = (tmp_path / "bundle.enc").read_bytes()

    # 1. Matching bundle bytes pass
    passed, errors = verify_record_and_anchor(receipt, bundle_bytes=bundle_bytes, quiet=True)
    assert passed is True
    assert len(errors) == 0

    # 2. Mutated bundle bytes fail
    mutated_bundle = bundle_bytes[:-1] + (b"\x00" if bundle_bytes[-1:] != b"\x00" else b"\x01")
    passed, errors = verify_record_and_anchor(receipt, bundle_bytes=mutated_bundle, quiet=True)
    assert passed is False
    assert any("Bundle ciphertext SHA-256 mismatch" in err for err in errors)


def test_milestone6_chains_and_abi():
    from pom.adapters.chain import CHAINS, get_contract_abi
    assert "amoy" in CHAINS
    assert CHAINS["amoy"]["chain_id"] == 80002
    assert "base-sepolia" in CHAINS
    assert CHAINS["base-sepolia"]["chain_id"] == 84532

    abi = get_contract_abi()
    func_names = [item.get("name") for item in abi if item.get("type") == "function"]
    assert "anchor" in func_names
    assert "isAnchored" in func_names


def test_milestone6_preflight_checks():
    from pom.adapters.chain import run_chain_preflight
    
    mock_w3 = MagicMock()
    mock_w3.is_connected.return_value = False
    mock_w3.provider.endpoint_uri = "http://fake-rpc"
    
    # 1. Unreachable RPC
    ok, err = run_chain_preflight(mock_w3, "0x123", "0x456", 80002, "amoy")
    assert ok is False
    assert "unreachable" in str(err)

    # 2. Chain ID mismatch
    mock_w3.is_connected.return_value = True
    mock_w3.eth.chain_id = 1  # mainnet instead of amoy (80002)
    ok, err = run_chain_preflight(mock_w3, "0x123", "0x456", 80002, "amoy")
    assert ok is False
    assert "Chain ID mismatch" in str(err)

    # 3. Zero balance
    mock_w3.eth.chain_id = 80002
    mock_w3.eth.get_code.return_value = b"\x60\x80"
    mock_w3.to_checksum_address.side_effect = lambda a: a
    mock_w3.eth.get_balance.return_value = 0
    ok, err = run_chain_preflight(mock_w3, "0x123", "0x456", 80002, "amoy")
    assert ok is False
    assert "0 balance" in str(err)


def test_milestone6_deploy_simulation_mode():
    from contracts.deploy import deploy_contract
    with patch.dict("os.environ", {"POM_PRIVATE_KEY": ""}):
        code = deploy_contract()
        assert code == 0


def test_milestone7_six_checks_and_tamper_exit_code_3(tmp_path):
    from pom.stage4_anchor import process_stage4
    from pom.verifier import verify_run_bundle

    stage3_data = {
        "run_timestamp_utc": "2026-09-06T12:00:00Z",
        "match_found": True,
        "matched_page_url": "https://linkedin.com/in/authentic",
        "matched_image_url": "https://media.authentic/img.jpg",
        "source_engine": "serpapi_exact",
        "face_match": {
            "arcface_cosine_similarity": 0.44,
            "adaface_cosine_similarity": 0.38,
            "arcface_threshold_used": 0.36,
            "adaface_threshold_used": 0.30,
            "decision_rule": "and_ensemble",
        },
        "candidates_tried": 1,
    }
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "input.jpg").write_bytes(b"image_bytes")

    stage1_data = {"arcface_embedding": [0.1] * 512}
    receipt = process_stage4(stage3_data, stage1_data, run_dir=tmp_path)
    record_file = tmp_path / "record.json"

    # 1. Six checks pass -> returns 0
    exit_code = verify_run_bundle(record_file, quiet=True)
    assert exit_code == 0

    # 2. Negative test with tamper -> returns 3
    disk_before = record_file.read_text(encoding="utf-8")
    tamper_exit_code = verify_run_bundle(
        record_file,
        tamper_field="match.similarity.score",
        tamper_value=0.9312,
        quiet=True,
    )
    assert tamper_exit_code == 3

    # 3. Asserts disk was NOT modified by tamper test (§13)
    disk_after = record_file.read_text(encoding="utf-8")
    assert disk_before == disk_after


def test_milestone8_calibrate_thresholds(tmp_path):
    from pom.calibrate import calibrate_thresholds, create_default_pairs_csv
    
    csv_file = tmp_path / "pairs.csv"
    create_default_pairs_csv(csv_file)
    assert csv_file.exists()

    cal_out = tmp_path / "calibration.json"
    result = calibrate_thresholds(pairs_csv_path=csv_file, output_json_path=cal_out, quiet=True)
    
    assert result["positive_pairs"] >= 5
    assert result["negative_pairs"] >= 25
    assert result["chosen_threshold"] > 0.0
    assert result["separation_gap"]["clean_separation"] is True
    assert result["evaluation"]["false_accept_rate_pct"] == 0.0
    assert result["evaluation"]["false_reject_rate_pct"] == 0.0
    assert cal_out.exists()




