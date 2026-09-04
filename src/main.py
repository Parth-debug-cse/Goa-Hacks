"""End-to-end orchestration for face search and blockchain verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from src.blockchain_client import get_record, register_record, verify_record
from src.content_capture import capture_post
from src.face_id import identify_face
from src.ipfs_upload import pin_to_ipfs
from src.web_search import best_social_match


def run(image_path):
    """Execute the happy-path pipeline from face match to on-chain verification.

    Args:
        image_path (str): Local path to the input face image.

    Returns:
        dict: Run summary containing at least record_id and content_hash.

    Raises:
        Exception: Propagates stage-specific failures from search, capture, upload, or chain calls.
    """
    embedding, _, _ = identify_face(image_path)
    best_match = best_social_match(image_path)

    content_hash, metadata_path, metadata = capture_post(best_match["url"], embedding)

    metadata_cid = pin_to_ipfs(metadata_path)
    screenshot_path = Path(metadata_path).with_name("screenshot.png")
    screenshot_cid = pin_to_ipfs(str(screenshot_path)) if screenshot_path.exists() else None

    bundle = {"metadata_cid": metadata_cid, "screenshot_cid": screenshot_cid}
    bundle_path = Path(metadata_path).with_name("metadata_bundle.json")
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    metadata_uri_cid = pin_to_ipfs(str(bundle_path))
    metadata_uri = f"ipfs://{metadata_uri_cid}"

    record_id = register_record(content_hash, best_match["url"], metadata_uri)
    verification = verify_record(record_id, content_hash)

    print(f"Matched URL: {best_match['url']}")
    print(f"Content hash: {content_hash.hex()}")
    print(f"Record ID: {record_id}")
    print(f"Verification result: {verification['matches']}")

    return {
        "record_id": record_id,
        "content_hash": content_hash,
        "metadata": metadata,
        "metadata_uri": metadata_uri,
    }


def run_tamper_demo(image_path: str) -> None:
    """Run the standard flow, then verify a deliberately tampered hash mismatch.

    Args:
        image_path (str): Local path to the input face image.

    Returns:
        None: Prints tamper demonstration results.

    Raises:
        Exception: Propagates errors from run() and contract calls.
    """
    result = run(image_path)
    record_id = result["record_id"]

    tampered_hash = hashlib.sha256(b"tampered").digest()
    on_chain_record = get_record(record_id)
    tampered_verification = verify_record(record_id, tampered_hash)

    print("EXPECTED: mismatch")
    print(f"RESULT: {tampered_verification['matches']}")
    print(f"ON-CHAIN RECORD: {on_chain_record}")


def main() -> None:
    """Parse CLI arguments and execute requested pipeline mode.

    Args:
        None.

    Returns:
        None.

    Raises:
        Exception: Propagates runtime errors from selected execution path.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Face → Web Match → Blockchain verification pipeline")
    parser.add_argument("image_path", help="Path to the local input image")
    parser.add_argument(
        "--tamper-demo",
        action="store_true",
        help="Run tamper-detection demo after successful record registration",
    )
    args = parser.parse_args()

    if args.tamper_demo:
        run_tamper_demo(args.image_path)
    else:
        run(args.image_path)


if __name__ == "__main__":
    main()
