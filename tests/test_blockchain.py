"""Tests for Polygon Amoy blockchain module (INV-4, INV-5, INV-7)."""

from unittest.mock import MagicMock, patch
from common.blockchain import (
    AMOY_CHAIN_ID,
    anchor_on_chain,
    get_rpc_url,
    verify_on_chain,
)


def test_anchor_on_chain_simulation_without_key():
    warnings = []
    record_hash = "0x" + "a" * 64
    ipfs_cid = "bafytest123"
    result = anchor_on_chain(record_hash, ipfs_cid, warnings=warnings)
    
    assert result["anchored"] is True
    assert result["simulated"] is True
    assert result["chain_id"] == AMOY_CHAIN_ID
    assert result["record_hash"] == record_hash
    assert result["tx_hash"].startswith("0x")
    assert any("blockchain_anchor_simulated" in w for w in warnings)


def test_verify_on_chain_with_mocked_rpc():
    mock_tx = {
        "blockNumber": "0xd9ff0a",
        "input": "0x5a18a994" + "a" * 64 + "62616679",
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x2222222222222222222222222222222222222222",
    }
    mock_receipt = {"status": "0x1"}

    with patch("common.blockchain._rpc_call") as mock_rpc:
        mock_rpc.side_effect = [mock_tx, mock_receipt]
        
        expected_hash = "0x" + "a" * 64
        valid, details, reason = verify_on_chain("0xtx123", expected_hash, "https://rpc.mock")
        
        assert valid is True
        assert details["status"] == "confirmed"
        assert "Confirmed" in reason


def test_verify_on_chain_detects_hash_mismatch():
    mock_tx = {
        "blockNumber": "0xd9ff0a",
        "input": "0x5a18a994" + "b" * 64,  # different hash
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x2222222222222222222222222222222222222222",
    }
    mock_receipt = {"status": "0x1"}

    with patch("common.blockchain._rpc_call") as mock_rpc:
        mock_rpc.side_effect = [mock_tx, mock_receipt]
        
        expected_hash = "0x" + "a" * 64
        valid, details, reason = verify_on_chain("0xtx123", expected_hash, "https://rpc.mock")
        
        assert valid is False
        assert "does not contain expected record hash" in reason
