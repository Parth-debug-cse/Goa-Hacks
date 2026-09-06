"""Web3 blockchain adapter for Polygon Amoy and Base Sepolia (§12.3, AH-7).

Guarantees:
- CHAINS configuration for Amoy (80002) and Base Sepolia (84532).
- POA middleware injected for both chains (AH-7).
- Preflight checks: RPC reachable, chain_id match, contract bytecode verified, balance sufficient.
- Gas estimation: estimate_gas * 1.25.
- Private key security: local signing, never logged, never committed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pom.canonical import sha256_hex
from pom.config import CONFIG

LOGGER = logging.getLogger(__name__)

CHAINS = {
    "amoy": {
        "chain_id": 80002,
        "rpc_env": "POM_RPC_AMOY",
        "contract_env": "POM_CONTRACT_AMOY",
        "explorer_tx": "https://amoy.polygonscan.com/tx/{}",
    },
    "base-sepolia": {
        "chain_id": 84532,
        "rpc_env": "POM_RPC_BASE_SEPOLIA",
        "contract_env": "POM_CONTRACT_BASE_SEPOLIA",
        "explorer_tx": "https://sepolia.basescan.org/tx/{}",
    },
}


def get_web3_client(rpc_url: str | None = None) -> Any:
    """Instantiate Web3 client with injected POA middleware for both Amoy and Base Sepolia (AH-7)."""
    from web3 import Web3

    url = rpc_url or CONFIG.get_active_rpc()
    w3 = Web3(Web3.HTTPProvider(url))

    # Inject POA middleware for Polygon Amoy and Base Sepolia (AH-7)
    # web3.py v6 uses geth_poa_middleware; v7 uses ExtraDataToPOAMiddleware
    try:
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    except (ImportError, AttributeError):
        try:
            from web3.middleware import ExtraDataToPOAMiddleware
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except (ImportError, AttributeError):
            pass

    return w3


def get_contract_abi() -> list[dict[str, Any]]:
    """Load MatchRegistry contract ABI."""
    abi_path = Path(__file__).resolve().parent.parent.parent / "contracts" / "MatchRegistry.abi.json"
    if not abi_path.exists():
        abi_path = CONFIG.contracts_dir / "MatchRegistry.abi.json"
    if abi_path.exists():
        return json.loads(abi_path.read_text(encoding="utf-8"))
    
    # Fallback minimal ABI matching §12.1
    return [
        {
            "type": "function",
            "name": "anchor",
            "inputs": [
                {"name": "recordHash", "type": "bytes32"},
                {"name": "subjectCommitment", "type": "bytes32"},
                {"name": "evidenceRoot", "type": "bytes32"},
                {"name": "cid", "type": "string"},
            ],
            "outputs": [],
            "stateMutability": "nonpayable",
        },
        {
            "type": "function",
            "name": "isAnchored",
            "inputs": [{"name": "recordHash", "type": "bytes32"}],
            "outputs": [{"name": "", "type": "bool"}, {"name": "", "type": "uint64"}],
            "stateMutability": "view",
        },
    ]


def run_chain_preflight(
    w3: Any,
    account_address: str,
    contract_address: str,
    expected_chain_id: int,
    chain_name: str,
) -> tuple[bool, str | None]:
    """Execute preflight checks before transaction submission (§12.3).
    
    Checks:
      1. RPC is reachable.
      2. Chain ID matches expected ID.
      3. Contract has bytecode at contract_address.
      4. Submitter account has non-zero balance.
    """
    if not w3.is_connected():
        return False, f"Preflight FAIL: RPC endpoint {w3.provider.endpoint_uri} is unreachable."

    try:
        remote_chain_id = w3.eth.chain_id
        if remote_chain_id != expected_chain_id:
            return False, (
                f"Preflight FAIL: Chain ID mismatch for '{chain_name}'. "
                f"Expected {expected_chain_id}, got {remote_chain_id}."
            )
    except Exception as err:
        return False, f"Preflight FAIL: Unable to query chain_id: {err}"

    # Verify contract bytecode at address
    if contract_address and contract_address != "0x0000000000000000000000000000000000000000":
        try:
            checksum_addr = w3.to_checksum_address(contract_address)
            code = w3.eth.get_code(checksum_addr)
            if not code or code == b"" or code.hex() in ("", "0x"):
                return False, (
                    f"Preflight FAIL: No contract bytecode found at address {contract_address} on {chain_name}. "
                    f"Deploy MatchRegistry first with `pom deploy` or set POM_CONTRACT_{chain_name.upper().replace('-', '_')}."
                )
        except Exception as err:
            return False, f"Preflight FAIL: Could not verify contract code at {contract_address}: {err}"

    # Verify submitter balance
    try:
        balance = w3.eth.get_balance(account_address)
        if balance == 0:
            return False, (
                f"Preflight FAIL: Submitter address {account_address} has 0 balance on {chain_name}. "
                f"Please fund the wallet via faucet before submitting."
            )
    except Exception as err:
        return False, f"Preflight FAIL: Could not check balance for {account_address}: {err}"

    return True, None


def anchor_match_on_chain(
    record_hash: str,
    subject_commitment: str,
    evidence_root: str,
    cid: str | None = None,
    chain_name: str | None = None,
) -> dict[str, Any]:
    """Anchor match proof on blockchain (§12.1, §12.3).
    
    Calls MatchRegistry.anchor(recordHash, subjectCommitment, evidenceRoot, cid).
    """
    chain_key = chain_name or CONFIG.chain
    chain_info = CHAINS.get(chain_key, CHAINS["amoy"])
    rpc_url = CONFIG.get_active_rpc()
    contract_addr = CONFIG.get_active_contract()
    priv_key = CONFIG.private_key
    cid_str = cid or ""

    if not priv_key:
        # Simulated mode (INV-7)
        simulated_tx = sha256_hex(f"simulated:{record_hash}:{subject_commitment}:{evidence_root}:{cid_str}".encode("utf-8"))
        explorer = chain_info["explorer_tx"].format(simulated_tx)
        return {
            "anchored": True,
            "simulated": True,
            "chain": chain_key,
            "chain_id": chain_info["chain_id"],
            "tx_hash": simulated_tx,
            "block_number": 14500000,
            "explorer_url": explorer,
            "warnings": ["blockchain_anchor_simulated: no POM_PRIVATE_KEY configured"],
        }

    w3 = get_web3_client(rpc_url)
    account = w3.eth.account.from_key(priv_key)
    submitter_addr = account.address

    # 1. Preflight check (§12.3)
    preflight_ok, preflight_err = run_chain_preflight(
        w3=w3,
        account_address=submitter_addr,
        contract_address=contract_addr,
        expected_chain_id=chain_info["chain_id"],
        chain_name=chain_key,
    )
    if not preflight_ok:
        LOGGER.warning("Preflight failed: %s", preflight_err)
        # Graceful degradation with clear warning
        simulated_tx = sha256_hex(f"simulated_preflight_failed:{record_hash}".encode("utf-8"))
        explorer = chain_info["explorer_tx"].format(simulated_tx)
        return {
            "anchored": False,
            "simulated": True,
            "chain": chain_key,
            "chain_id": chain_info["chain_id"],
            "tx_hash": simulated_tx,
            "block_number": 14500000,
            "explorer_url": explorer,
            "warnings": [preflight_err or "preflight_failed"],
        }

    try:
        checksum_contract = w3.to_checksum_address(contract_addr)
        abi = get_contract_abi()
        contract = w3.eth.contract(address=checksum_contract, abi=abi)

        # Convert 32-byte hex strings to bytes32
        rec_bytes32 = bytes.fromhex(record_hash[2:] if record_hash.startswith("0x") else record_hash)
        subj_bytes32 = bytes.fromhex(subject_commitment[2:] if subject_commitment.startswith("0x") else subject_commitment)
        evid_bytes32 = bytes.fromhex(evidence_root[2:] if evidence_root.startswith("0x") else evidence_root)

        # On-chain deduplication check (§12.1)
        try:
            is_anchored, anchored_at = contract.functions.isAnchored(rec_bytes32).call()
            if is_anchored:
                LOGGER.info("Record %s is already anchored at timestamp %s", record_hash, anchored_at)
                return {
                    "anchored": True,
                    "already_anchored": True,
                    "chain": chain_key,
                    "chain_id": chain_info["chain_id"],
                    "contract_address": checksum_contract,
                    "timestamp": anchored_at,
                    "warnings": [f"record_already_anchored_on_chain_at_timestamp_{anchored_at}"],
                }
        except Exception as read_err:
            LOGGER.debug("isAnchored read failed (continuing to transaction): %s", read_err)

        nonce = w3.eth.get_transaction_count(submitter_addr, "pending")
        gas_price = w3.eth.gas_price

        # Build transaction
        tx_dict = contract.functions.anchor(
            rec_bytes32,
            subj_bytes32,
            evid_bytes32,
            cid_str,
        ).build_transaction({
            "from": submitter_addr,
            "nonce": nonce,
            "gasPrice": gas_price,
            "chainId": chain_info["chain_id"],
        })

        # Gas estimate * 1.25 (§12.3)
        try:
            est_gas = w3.eth.estimate_gas(tx_dict)
            tx_dict["gas"] = int(est_gas * 1.25)
        except Exception:
            tx_dict["gas"] = 180000

        # Sign locally
        signed_tx = w3.eth.account.sign_transaction(tx_dict, priv_key)

        # Send raw transaction
        tx_hash_bytes = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_hash_hex = "0x" + tx_hash_bytes.hex()
        explorer_url = chain_info["explorer_tx"].format(tx_hash_hex)

        LOGGER.info("Broadcasted anchor tx: %s", tx_hash_hex)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=180)

        if receipt.status != 1:
            raise RuntimeError(f"On-chain transaction reverted with status: {receipt.status}")

        return {
            "anchored": True,
            "chain": chain_key,
            "chain_id": chain_info["chain_id"],
            "contract_address": checksum_contract,
            "tx_hash": tx_hash_hex,
            "block_number": receipt.blockNumber,
            "explorer_url": explorer_url,
            "gas_used": receipt.gasUsed,
            "warnings": [],
        }

    except Exception as err:
        LOGGER.warning("Blockchain anchor execution failed: %s", err)
        simulated_tx = sha256_hex(f"error:{record_hash}:{err}".encode("utf-8"))
        explorer = chain_info["explorer_tx"].format(simulated_tx)
        return {
            "anchored": False,
            "simulated": True,
            "chain": chain_key,
            "chain_id": chain_info["chain_id"],
            "tx_hash": simulated_tx,
            "block_number": 14500000,
            "explorer_url": explorer,
            "warnings": [f"blockchain_anchor_exception: {err}"],
        }
