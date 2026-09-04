"""Pinata IPFS upload helper."""

from __future__ import annotations

import os

import requests


def pin_to_ipfs(file_path: str) -> str:
    """Upload a file to Pinata and return its CID.

    Args:
        file_path (str): Local file path to upload.

    Returns:
        str: IPFS CID returned by Pinata.

    Raises:
        KeyError: If PINATA_JWT is missing from environment.
        RuntimeError: If Pinata upload fails.
    """
    jwt = os.environ["PINATA_JWT"]

    try:
        with open(file_path, "rb") as file_handle:
            response = requests.post(
                "https://api.pinata.cloud/pinning/pinFileToIPFS",
                headers={"Authorization": "Bearer " + jwt},
                files={"file": file_handle},
                timeout=30,
            )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Pinata upload failed for {file_path}: {error}") from error

    payload = response.json()
    cid = payload.get("IpfsHash")
    if not cid:
        raise RuntimeError("Pinata response missing IpfsHash")
    return cid
