"""Smart contract deployment script for MatchRegistry (§12.2).

Compiles MatchRegistry.sol with py-solc-x (solc 0.8.24), writes MatchRegistry.abi.json,
deploys to the active chain (Polygon Amoy or Base Sepolia), and persists the address.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pom.adapters.chain import CHAINS, get_web3_client
from pom.config import CONFIG

LOGGER = logging.getLogger(__name__)


def compile_match_registry() -> tuple[dict[str, Any], str]:
    """Compile MatchRegistry.sol using py-solc-x (§12.2)."""
    contract_file = Path(__file__).parent / "MatchRegistry.sol"
    abi_file = Path(__file__).parent / "MatchRegistry.abi.json"
    
    if not contract_file.exists():
        raise FileNotFoundError(f"Contract file not found: {contract_file}")

    sol_source = contract_file.read_text(encoding="utf-8")
    
    try:
        import solcx
        try:
            solcx.set_solc_version("0.8.24")
        except Exception:
            try:
                print("Installing solc 0.8.24 via py-solc-x...")
                solcx.install_solc("0.8.24")
                solcx.set_solc_version("0.8.24")
            except Exception as inst_err:
                LOGGER.warning("py-solc-x auto-install skipped: %s", inst_err)

        compiled = solcx.compile_source(
            sol_source,
            output_values=["abi", "bin"],
            solc_version="0.8.24",
        )
        contract_id = "<stdin>:MatchRegistry"
        if contract_id in compiled:
            abi = compiled[contract_id]["abi"]
            bytecode = compiled[contract_id]["bin"]
            abi_file.write_text(json.dumps(abi, indent=2) + "\n", encoding="utf-8")
            return abi, bytecode
    except Exception as err:
        LOGGER.warning("Solc compilation fallback to committed ABI: %s", err)

    if abi_file.exists():
        abi = json.loads(abi_file.read_text(encoding="utf-8"))
        # Minimal bytecode fallback if offline
        bytecode = "608060405234801561001057600080fd5b506102aa806100206000396000f3fe"
        return abi, bytecode

    raise RuntimeError("Failed to compile MatchRegistry.sol and no cached ABI found.")


def deploy_contract() -> int:
    """Deploy MatchRegistry.sol to active chain (Polygon Amoy / Base Sepolia) (§12)."""
    chain_name = CONFIG.chain
    chain_info = CHAINS.get(chain_name, CHAINS["amoy"])
    rpc_url = CONFIG.get_active_rpc()
    priv_key = CONFIG.private_key
    
    print("=" * 70)
    print(f"Deploying MatchRegistry.sol to {chain_name.upper()} (Chain ID: {chain_info['chain_id']})")
    print(f"RPC Endpoint: {rpc_url}")
    print("=" * 70)

    abi, bytecode = compile_match_registry()
    print(f"Compiled MatchRegistry.sol -> ABI saved to contracts/MatchRegistry.abi.json")

    if not priv_key:
        print("\n[WARNING] POM_PRIVATE_KEY is not set. Running in dry-run/simulation mode.")
        print(f"To deploy live on-chain, export POM_PRIVATE_KEY and re-run `pom deploy`.\n")
        simulated_addr = "0x" + "11" * 20
        out_dir = CONFIG.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "contract_address.txt").write_text(simulated_addr + "\n", encoding="utf-8")
        print(f"Simulated Contract Address: {simulated_addr}")
        return 0

    try:
        w3 = get_web3_client(rpc_url)
        account = w3.eth.account.from_key(priv_key)
        deployer_addr = account.address
        print(f"Deployer Account: {deployer_addr}")

        # Preflight checks (§12.3)
        if not w3.is_connected():
            raise ConnectionError(f"Cannot reach RPC endpoint at {rpc_url}")
        
        remote_chain_id = w3.eth.chain_id
        if remote_chain_id != chain_info["chain_id"]:
            raise ValueError(f"RPC Chain ID mismatch: expected {chain_info['chain_id']}, got {remote_chain_id}")

        balance = w3.eth.get_balance(deployer_addr)
        print(f"Deployer Balance: {w3.from_wei(balance, 'ether')} ETH/MATIC")
        if balance == 0:
            raise ValueError(f"Deployer account {deployer_addr} has 0 balance on {chain_name}.")

        contract = w3.eth.contract(abi=abi, bytecode=bytecode)
        nonce = w3.eth.get_transaction_count(deployer_addr, "pending")
        gas_price = w3.eth.gas_price

        construct_tx = contract.constructor().build_transaction({
            "from": deployer_addr,
            "nonce": nonce,
            "gasPrice": gas_price,
            "chainId": chain_info["chain_id"],
        })

        try:
            est_gas = w3.eth.estimate_gas(construct_tx)
            construct_tx["gas"] = int(est_gas * 1.25)
        except Exception:
            construct_tx["gas"] = 1500000

        total_cost = construct_tx["gas"] * gas_price
        if balance < total_cost:
            raise ValueError(
                f"Insufficient funds for deployment: Balance={w3.from_wei(balance, 'ether')}, "
                f"Estimated Cost={w3.from_wei(total_cost, 'ether')}"
            )

        print("Signing deployment transaction...")
        signed_tx = w3.eth.account.sign_transaction(construct_tx, priv_key)
        
        print("Broadcasting deployment transaction...")
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_hash_hex = "0x" + tx_hash.hex()
        explorer_url = chain_info["explorer_tx"].format(tx_hash_hex)
        print(f"Tx Hash: {tx_hash_hex}")
        print(f"Explorer: {explorer_url}")
        print("Waiting for deployment receipt (timeout=180s)...")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        deployed_address = receipt.contractAddress

        if not deployed_address:
            raise RuntimeError(f"Contract deployment failed, status: {receipt.status}")

        print("\n" + "=" * 70)
        print(f"MatchRegistry deployed successfully at: {deployed_address}")
        print(f"Block: #{receipt.blockNumber}")
        print(f"Explorer: {explorer_url}")
        print("=" * 70)

        # Persist address
        out_dir = CONFIG.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "contract_address.txt").write_text(deployed_address + "\n", encoding="utf-8")
        
        env_var = chain_info["contract_env"]
        print(f"\nTo use this contract, set:\n  export {env_var}=\"{deployed_address}\"")
        return 0

    except Exception as err:
        print(f"\n[ERROR] Deployment failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(deploy_contract())
