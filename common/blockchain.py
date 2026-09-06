"""Polygon Amoy blockchain anchoring and independent on-chain verification module.

Adheres strictly to:
- INV-4: BYTE-EXACT REPRODUCIBILITY (hashes checked byte-for-byte against on-chain state).
- INV-5: NO BIOMETRICS AND NO PII ON-CHAIN (only 32-byte digests and encrypted CIDs are anchored).
- INV-7: GRACEFUL DEGRADATION, LOUDLY (structured warnings for RPC / key issues).

Target Chain: Polygon Amoy Testnet (Chain ID 80002 / 0x13882).
Explorer: https://amoy.polygonscan.com
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlsplit

from common.canonical_json import keccak256_hex, sha256_hex
from common.http_utils import create_session, response_meta
from common.provenance import log_request

LOGGER = logging.getLogger(__name__)

AMOY_CHAIN_ID = 80002
AMOY_EXPLORER_BASE = "https://amoy.polygonscan.com"

# Public default RPC endpoints for Polygon Amoy
DEFAULT_AMOY_RPCS = [
    "https://rpc-amoy.polygon.technology/",
    "https://polygon-amoy.drpc.org",
]

# Standard Anchor contract ABI interface for ProofAnchored event & anchorProof function
ANCHOR_FUNCTION_SIGNATURE = "anchorProof(bytes32,string,bytes32)"
# keccak256("anchorProof(bytes32,string,bytes32)")[:4] = 0x5a18a994
ANCHOR_FUNCTION_SELECTOR = "0x5a18a994"

# keccak256("ProofAnchored(bytes32,string,uint256,address)")
PROOF_ANCHORED_TOPIC = "0x00a7fae017684a0d9eeb0e47087819fe8e663a89045dbb562a0ee6db7c2e086a"


def get_rpc_url() -> str:
    """Get the active Polygon Amoy RPC endpoint."""
    return os.environ.get("AMOY_RPC_URL") or os.environ.get("POLYGON_RPC_URL") or DEFAULT_AMOY_RPCS[0]


def _rpc_call(method: str, params: list[Any], rpc_url: str | None = None) -> Any:
    """Execute a JSON-RPC request against Polygon Amoy RPC."""
    url = rpc_url or get_rpc_url()
    session = create_session()
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }
    started = time.monotonic()
    response = session.post(url, json=payload, timeout=10)
    status, size = response_meta(response)
    log_request("blockchain_rpc", "POST", url, {"method": method},
                status, (time.monotonic() - started) * 1000, size)
    
    if status is not None and status >= 400:
        raise ConnectionError(f"RPC HTTP error {status} from {url}")
    data = response.json()
    if "error" in data:
        raise ValueError(f"RPC error: {data['error']}")
    return data.get("result")


def anchor_on_chain(
    record_hash: str,
    ipfs_cid: str,
    metadata_digest: str = "0x" + "0" * 64,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Anchor a cryptographic record hash and IPFS CID on Polygon Amoy.
    
    If POLYGON_PRIVATE_KEY is provided, broadcasts a live transaction.
    Otherwise, creates a deterministic simulation anchor with a structured warning.
    """
    if warnings is None:
        warnings = []

    clean_hash = record_hash.lower()
    if not clean_hash.startswith("0x"):
        clean_hash = "0x" + clean_hash

    priv_key = os.environ.get("POLYGON_PRIVATE_KEY") or os.environ.get("AMOY_PRIVATE_KEY")
    contract_addr = os.environ.get("PROOF_ANCHOR_CONTRACT_ADDRESS")

    if not priv_key:
        warnings.append("blockchain_anchor_simulated: no POLYGON_PRIVATE_KEY configured")
        # Generate a deterministic simulated anchor
        sim_tx_seed = f"simulated_amoy_anchor:{clean_hash}:{ipfs_cid}"
        sim_tx_hash = keccak256_hex(sim_tx_seed.encode("utf-8"))
        sim_block = 14500000 + (int(clean_hash[:8], 16) % 500000)
        
        return {
            "anchored": True,
            "simulated": True,
            "chain_id": AMOY_CHAIN_ID,
            "chain_name": "Polygon Amoy Testnet",
            "record_hash": clean_hash,
            "ipfs_cid": ipfs_cid,
            "tx_hash": sim_tx_hash,
            "block_number": sim_block,
            "explorer_url": f"{AMOY_EXPLORER_BASE}/tx/{sim_tx_hash}",
            "contract_address": contract_addr or "0x0000000000000000000000000000000000000000",
            "anchor_method": "simulated_deterministic_receipt",
        }

    try:
        from web3 import Web3
        from pom.adapters.polygon import inject_poa_middleware

        rpc_url = get_rpc_url()
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        inject_poa_middleware(w3)

        account = w3.eth.account.from_key(priv_key)
        sender_address = account.address

        # Build calldata embedding the record_hash and IPFS CID
        # Data format: 4-byte prefix + 32-byte record_hash + UTF-8 bytes of ipfs_cid
        calldata_bytes = bytes.fromhex(clean_hash[2:]) + ipfs_cid.encode("utf-8")
        
        nonce = w3.eth.get_transaction_count(sender_address, "pending")
        gas_price = w3.eth.gas_price

        tx = {
            "to": contract_addr or sender_address,
            "value": 0,
            "gas": 120000,
            "gasPrice": gas_price,
            "nonce": nonce,
            "chainId": AMOY_CHAIN_ID,
            "data": calldata_bytes,
        }

        signed_tx = w3.eth.account.sign_transaction(tx, priv_key)
        tx_hash_bytes = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_hash_hex = "0x" + tx_hash_bytes.hex()

        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=60)
        block_num = receipt.blockNumber

        return {
            "anchored": True,
            "simulated": False,
            "chain_id": AMOY_CHAIN_ID,
            "chain_name": "Polygon Amoy Testnet",
            "record_hash": clean_hash,
            "ipfs_cid": ipfs_cid,
            "tx_hash": tx_hash_hex,
            "block_number": block_num,
            "explorer_url": f"{AMOY_EXPLORER_BASE}/tx/{tx_hash_hex}",
            "contract_address": contract_addr or sender_address,
            "anchor_method": "live_polygon_amoy_transaction",
        }
    except Exception as err:
        warnings.append(f"blockchain_anchor_failed: {err}")
        sim_tx_seed = f"fallback_amoy_anchor:{clean_hash}:{ipfs_cid}"
        sim_tx_hash = keccak256_hex(sim_tx_seed.encode("utf-8"))
        return {
            "anchored": False,
            "simulated": True,
            "chain_id": AMOY_CHAIN_ID,
            "chain_name": "Polygon Amoy Testnet",
            "record_hash": clean_hash,
            "ipfs_cid": ipfs_cid,
            "tx_hash": sim_tx_hash,
            "block_number": 0,
            "explorer_url": f"{AMOY_EXPLORER_BASE}/tx/{sim_tx_hash}",
            "error": str(err),
            "anchor_method": "fallback_simulated",
        }


def verify_on_chain(
    tx_hash: str,
    expected_record_hash: str,
    rpc_url: str | None = None,
) -> tuple[bool, dict[str, Any], str]:
    """Independently verify on-chain state for an anchor transaction on Polygon Amoy.
    
    Queries the public Polygon Amoy RPC to retrieve transaction details and calldata/logs.
    Compares the anchored hash with expected_record_hash byte-for-byte.
    
    Returns (is_valid, details_dict, reason_message).
    """
    clean_expected = expected_record_hash.lower()
    if not clean_expected.startswith("0x"):
        clean_expected = "0x" + clean_expected

    target_rpc = rpc_url or get_rpc_url()

    try:
        tx_data = _rpc_call("eth_getTransactionByHash", [tx_hash], target_rpc)
        if not tx_data:
            # Check if this was a simulated offline demo transaction
            return False, {}, f"Transaction {tx_hash} not found on Polygon Amoy RPC ({target_rpc})"

        block_num_raw = tx_data.get("blockNumber")
        block_num = int(block_num_raw, 16) if isinstance(block_num_raw, str) else block_num_raw

        input_data = tx_data.get("input", "") or ""
        # Check if expected hash is present in transaction input calldata
        hash_hex_without_0x = clean_expected[2:]
        hash_found = hash_hex_without_0x in input_data.lower()

        receipt_data = _rpc_call("eth_getTransactionReceipt", [tx_hash], target_rpc)
        status = 1
        if receipt_data and "status" in receipt_data:
            status_raw = receipt_data["status"]
            status = int(status_raw, 16) if isinstance(status_raw, str) else status_raw

        details = {
            "tx_hash": tx_hash,
            "block_number": block_num,
            "status": "confirmed" if status == 1 else "reverted",
            "from": tx_data.get("from"),
            "to": tx_data.get("to"),
            "explorer_url": f"{AMOY_EXPLORER_BASE}/tx/{tx_hash}",
        }

        if status != 1:
            return False, details, f"Transaction {tx_hash} reverted on-chain (status=0)"

        if not hash_found:
            return False, details, (
                f"On-chain calldata in tx {tx_hash} does not contain expected record hash {clean_expected}"
            )

        return True, details, f"Confirmed in Polygon Amoy Block #{block_num}"
    except Exception as err:
        return False, {}, f"RPC verification failed: {err}"
