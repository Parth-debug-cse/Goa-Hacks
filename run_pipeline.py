#!/usr/bin/env python3
"""Consent-gated Stage 1 -> Stage 2 -> Stage 3 demo runner."""

from __future__ import annotations

import argparse
import json
import logging

from stage1_face import process_image
from stage2_search import process_search
from stage3_verify import process_verification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("--consent-confirmed", action="store_true", required=True,
                        help="Required acknowledgement that the photo is a consenting team member.")
    parser.add_argument("--adaface-root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
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
        print(json.dumps({"match_found": False, "source_photo_note": stage1, "candidates_tried": 0,
                          "candidates_rejected": [], "warnings": ["stage1_quality_or_embedding_failed"]}, sort_keys=True))
        return 1
    candidates, warnings = process_search(args.photo)
    result = process_verification(candidates, stage1)
    result["warnings"] = warnings + result.get("warnings", [])
    print(json.dumps(result, sort_keys=True))
    return 0 if result["match_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
