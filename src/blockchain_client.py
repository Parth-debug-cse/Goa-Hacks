"""Polygon Amoy contract interaction client."""

from __future__ import annotations

import json
import os
from pathlib import Path

from web3 import Web3

CHAIN_ID = 80002
ABI_PATH = Path(__file__).resolve().parents[1] / "contracts" / "abi.json"


def get_web3() -> Web3:
    """Create a Web3 client connected to Polygon Amoy.

    Returns:
        Web3: Connected Web3 instance.

    Raises:
        KeyError: If POLYGON_AMOY_RPC_URL is missing.
        ConnectionError: If the RPC endpoint is unreachable.
    """
    rpc_url = os.environ["POLYGON_AMOY_RPC_URL"]
    web3 = Web3(Web3.HTTPProvider(rpc_url))
    if not web3.is_connected():
        raise ConnectionError(f"Unable to connect to Polygon RPC endpoint: {rpc_url}")
    return web3


def get_contract():
    """Load deployed contract using ABI and configured address.

    Returns:
        web3.contract.Contract: Contract instance.

    Raises:
        FileNotFoundError: If ABI JSON file is missing.
        KeyError: If CONTRACT_ADDRESS is missing.
    """
    web3 = get_web3()
    address = os.environ["CONTRACT_ADDRESS"]

    with open(ABI_PATH, "r", encoding="utf-8") as abi_file:
        abi = json.load(abi_file)

    return web3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)


def register_record(content_hash: bytes, source_url: str, metadata_uri: str) -> int:
    """Submit a new record transaction and return emitted record ID.

    Args:
        content_hash (bytes): 32-byte hash of captured content.
        source_url (str): Matched source URL.
        metadata_uri (str): IPFS URI containing metadata bundle.

    Returns:
        int: Record ID emitted by RecordRegistered event.

    Raises:
        KeyError: If DEPLOYER_PRIVATE_KEY is missing.
        ValueError: If transaction receipt does not include expected event.
    """
    web3 = get_web3()
    contract = get_contract()

    private_key = os.environ["DEPLOYER_PRIVATE_KEY"]
    account = web3.eth.account.from_key(private_key)

    tx = contract.functions.registerRecord(content_hash, source_url, metadata_uri).build_transaction(
        {
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(account.address),
            "chainId": CHAIN_ID,
            "gasPrice": web3.eth.gas_price,
        }
    )

    signed = account.sign_transaction(tx)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    events = contract.events.RecordRegistered().process_receipt(receipt)
    if not events:
        raise ValueError("RecordRegistered event not found in transaction receipt")

    return int(events[0]["args"]["recordId"])


def verify_record(record_id: int, content_hash: bytes) -> dict:
    """Verify whether a provided hash matches the on-chain record.

    Args:
        record_id (int): Record identifier.
        content_hash (bytes): Hash to verify.

    Returns:
        dict: Verification result with keys {record_id, matches}.

    Raises:
        ValueError: If contract call fails due to invalid input.
    """
    contract = get_contract()
    matches = contract.functions.verifyRecord(record_id, content_hash).call()
    return {"record_id": record_id, "matches": bool(matches)}


def get_record(record_id: int) -> dict:
    """Fetch a stored record from the contract's public mapping getter.

    Args:
        record_id (int): Record identifier.

    Returns:
        dict: Record fields {content_hash, source_url, metadata_uri, timestamp, submitter}.

    Raises:
        ValueError: If the contract call returns an unexpected payload shape.
    """
    contract = get_contract()
    record = contract.functions.records(record_id).call()

    if not isinstance(record, (list, tuple)) or len(record) < 5:
        raise ValueError(f"Unexpected record payload: {record}")

    return {
        "content_hash": record[0],
        "source_url": record[1],
        "metadata_uri": record[2],
        "timestamp": int(record[3]),
        "submitter": record[4],
    }
