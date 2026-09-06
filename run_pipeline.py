#!/usr/bin/env python3
"""Consent-gated Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 pipeline orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from stage1_face import process_image
from stage2_search import process_search
from stage3_verify import process_verification
from stage4_anchor import process_anchoring


def main() -> int:
    parser = argparse.ArgumentParser(description="Consent-gated Face -> Search -> Verify -> Anchor pipeline.")
    parser.add_argument("photo", help="Path to reference photo of consenting person")
    parser.add_argument(
        "--consent-confirmed",
        action="store_true",
        required=True,
        help="Required acknowledgement that the photo is a consenting team member (INV-6).",
    )
    parser.add_argument("--adaface-root", help="Path to AdaFace repo root")
    parser.add_argument(
        "--output",
        "-o",
        default="anchor_receipt.json",
        help="Output file for anchor receipt (default: anchor_receipt.json)",
    )
    parser.add_argument(
        "--no-anchor",
        action="store_true",
        help="Skip Stage 4 blockchain anchoring (output raw Stage 3 handoff JSON only)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.photo):
        print(f"Error: Refusing directory input '{args.photo}'. Batch processing is forbidden (INV-6).", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
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
        print(
            json.dumps(
                {
                    "match_found": False,
                    "source_photo_note": stage1,
                    "candidates_tried": 0,
                    "candidates_rejected": [],
                    "warnings": ["stage1_quality_or_embedding_failed"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 1

    candidates, warnings = process_search(args.photo)
    result = process_verification(candidates, stage1)
    result["warnings"] = warnings + result.get("warnings", [])

    if not result.get("match_found"):
        print(json.dumps(result, sort_keys=True, indent=2))
        return 2

    if args.no_anchor:
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    # Stage 4: Blockchain Anchoring
    receipt = process_anchoring(result, output_path=args.output)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
