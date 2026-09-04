"""Compile and deploy PostVerification contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from solcx import compile_standard, install_solc
from web3 import Web3

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "PostVerification.sol"
ABI_PATH = ROOT / "contracts" / "abi.json"


def deploy_contract() -> str:
    """Compile and deploy the Solidity contract to Polygon Amoy-compatible RPC.

    Returns:
        str: Deployed contract address.

    Raises:
        KeyError: If required environment variables are missing.
        RuntimeError: If deployment transaction fails.
    """
    load_dotenv()
    if ABI_PATH.exists():
        print(
            "abi.json already exists — deploying will not overwrite it; "
            "delete it first if you want a fresh ABI"
        )

    install_solc("0.8.20")

    source = CONTRACT_PATH.read_text(encoding="utf-8")
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"PostVerification.sol": {"content": source}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode"]
                    }
                }
            },
        },
        solc_version="0.8.20",
    )

    contract_data = compiled["contracts"]["PostVerification.sol"]["PostVerification"]
    abi = contract_data["abi"]
    bytecode = contract_data["evm"]["bytecode"]["object"]

    if not ABI_PATH.exists():
        ABI_PATH.write_text(json.dumps(abi, indent=2), encoding="utf-8")

    web3 = Web3(Web3.HTTPProvider(os.environ["POLYGON_AMOY_RPC_URL"]))
    if not web3.is_connected():
        raise RuntimeError("Failed to connect to Polygon Amoy RPC")

    account = web3.eth.account.from_key(os.environ["DEPLOYER_PRIVATE_KEY"])
    contract = web3.eth.contract(abi=abi, bytecode=bytecode)

    tx = contract.constructor().build_transaction(
        {
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(account.address),
            "chainId": 80002,
            "gasPrice": web3.eth.gas_price,
        }
    )

    signed = account.sign_transaction(tx)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.contractAddress


if __name__ == "__main__":
    deployed_address = deploy_contract()
    print(f"Contract deployed at: {deployed_address}")
