#!/usr/bin/env python3
"""Unified CLI entrypoint for Proof-of-Match (pom).

Usage:
  pom run <photo> --consent-confirmed [--output anchor_receipt.json]
  pom verify [anchor_receipt.json] [--tamper-demo]

Invariants:
- INV-1: NO FABRICATED RESULTS. Exits 2 if search returns nothing. Never invent a match.
- INV-2: PROVENANCE OR IT DOESN'T EXIST. Enforced via Stage 2/3 provenance assertions.
- INV-3: NO LLM IN THE BIOMETRIC PATH. ArcFace + AdaFace cosine similarity only.
- INV-4: BYTE-EXACT REPRODUCIBILITY. Deterministic canonical JSON and hashes.
- INV-5: NO BIOMETRICS AND NO PII ON-CHAIN OR ON PUBLIC IPFS. Encrypted bundles.
- INV-6: CONSENT GATE. Must have --consent-confirmed. Rejects directories and batch mode.
- INV-7: GRACEFUL DEGRADATION, LOUDLY. Structured warnings on missing keys/RPCs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from verify import main as verify_main


def cmd_run(args: argparse.Namespace) -> int:
    """Run the complete end-to-end pipeline."""
    # INV-6: Consent gate and strict single-photo verification
    if not args.consent_confirmed:
        print(
            "Error: --consent-confirmed is REQUIRED. This pipeline is only for consenting team members.",
            file=sys.stderr,
        )
        return 1

    if os.path.isdir(args.photo):
        print(
            f"Error: Refusing directory input '{args.photo}'. Batch processing is forbidden (INV-6).",
            file=sys.stderr,
        )
        return 1

    if not os.path.isfile(args.photo):
        print(f"Error: Photo file not found: '{args.photo}'", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    LOGGER = logging.getLogger("pom")

    from stage1_face import process_image
    from stage2_search import process_search
    from stage3_verify import process_verification
    from stage4_anchor import process_anchoring

    LOGGER.info("Step 1: Face detection, quality gate, and dual embeddings...")
    stage1 = process_image(args.photo, args.adaface_root)
    arcface_embedding = stage1.get("arcface_embedding")
    adaface_embedding = stage1.get("adaface_embedding")

    if (
        not stage1.get("quality_passed")
        or arcface_embedding is None
        or adaface_embedding is None
        or not hasattr(arcface_embedding, "__len__")
        or not hasattr(adaface_embedding, "__len__")
        or len(arcface_embedding) == 0
        or len(adaface_embedding) == 0
    ):
        LOGGER.error("Stage 1 quality gate or embedding failed: %s", stage1.get("reason", "unknown"))
        output_payload = {
            "match_found": False,
            "source_photo_note": stage1,
            "candidates_tried": 0,
            "candidates_rejected": [],
            "warnings": ["stage1_quality_or_embedding_failed"],
        }
        print(json.dumps(output_payload, sort_keys=True, indent=2))
        return 1

    LOGGER.info("Step 2: Genuine reverse-image search and hop-2 identity pivot...")
    candidates, warnings = process_search(args.photo)

    LOGGER.info("Step 3: Candidate page fetching, biometric verification, and PDL enrichment...")
    stage3 = process_verification(candidates, stage1)
    stage3["warnings"] = warnings + stage3.get("warnings", [])

    if not stage3.get("match_found"):
        LOGGER.warning("No verified match found across %d candidates (INV-1).", stage3.get("candidates_tried", 0))
        print(json.dumps(stage3, sort_keys=True, indent=2))
        return 2

    LOGGER.info("Step 4: Cryptographic proof generation and Polygon Amoy anchoring...")
    receipt = process_anchoring(stage3, output_path=args.output)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pom",
        description="Proof-of-Match (pom): Tamper-Evident Biometric Face-to-Blockchain Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run the full pipeline on a photo of a consenting person")
    run_parser.add_argument("photo", help="Path to single reference photo")
    run_parser.add_argument(
        "--consent-confirmed",
        action="store_true",
        required=True,
        help="Mandatory acknowledgement that the photo is a consenting person (INV-6)",
    )
    run_parser.add_argument("--adaface-root", help="Path to AdaFace repo root")
    run_parser.add_argument(
        "--output",
        "-o",
        default="anchor_receipt.json",
        help="Path to save anchor receipt JSON (default: anchor_receipt.json)",
    )

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Independently verify an anchor receipt or run tamper demo")
    verify_parser.add_argument(
        "receipt",
        nargs="?",
        default="anchor_receipt.json",
        help="Path to anchor receipt JSON file (default: anchor_receipt.json)",
    )
    verify_parser.add_argument("--rpc-url", help="Custom Polygon Amoy RPC endpoint URL")
    verify_parser.add_argument(
        "--tamper-demo",
        action="store_true",
        help="Demonstrate tamper-evidence by altering 1 character before verification",
    )
    verify_parser.add_argument(
        "--field",
        default="similarity",
        choices=["similarity", "url", "timestamp"],
        help="Field to mutate in tamper-demo mode",
    )
    verify_parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "verify":
        from pom.verifier import main as verifier_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return verifier_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
