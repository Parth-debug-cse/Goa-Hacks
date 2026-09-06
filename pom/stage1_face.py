"""Stage 1: Face detection, quality gate, and dual embeddings (§3).

Guarantees:
- INV-3: NO LLM IN THE BIOMETRIC PATH. ArcFace + AdaFace only.
- AH-7: InsightFace face.normed_embedding normalization consistency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

import stage1_face as legacy_stage1
from pom.config import CONFIG

LOGGER = logging.getLogger(__name__)


def process_stage1(
    photo_path: str,
    run_dir: Path | None = None,
    adaface_root: str | None = None,
) -> dict[str, Any]:
    """Execute Stage 1 detection, quality gate, and dual embeddings."""
    result = legacy_stage1.process_image(photo_path, adaface_root)

    if run_dir:
        stage1_file = run_dir / "stage1.json"
        stage1_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Save face crop if detection succeeded
        if result.get("quality_passed"):
            try:
                img = Image.open(photo_path)
                evidence_crop = run_dir / "evidence" / "face_crop.png"
                img.save(evidence_crop)
            except Exception as err:
                LOGGER.warning("Could not save face_crop evidence: %s", err)

    return result
