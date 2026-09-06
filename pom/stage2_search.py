"""Stage 2: Reverse-image search and hop-2 identity pivot (§3).

Guarantees:
- INV-1: NO FABRICATED RESULTS.
- INV-2: PROVENANCE OR IT DOESN'T EXIST.
- AH-6: Search adapters in pom/adapters/
- AH-7: Retired Microsoft Bing endpoints deleted, routed through SerpApi.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from stage2_search import CandidateURL, process_search

LOGGER = logging.getLogger(__name__)


def process_stage2(
    photo_path: str,
    run_dir: Path | None = None,
    timeout_seconds: float = 15.0,
) -> tuple[list[CandidateURL], list[str]]:
    """Execute Stage 2 multi-engine reverse search and hop-2 identity pivot."""
    candidates, warnings = process_search(photo_path, timeout_seconds=timeout_seconds)

    if run_dir:
        candidates_dict = [
            {
                "url": c.url,
                "title": c.title,
                "source_engine": c.source_engine,
                "match_confidence_hint": c.match_confidence_hint,
                "provenance_id": c.provenance_id,
                "search_hop": c.search_hop,
                "discovered_via": c.discovered_via,
            }
            for c in candidates
        ]
        stage2_data = {
            "candidates_found": len(candidates),
            "candidates": candidates_dict,
            "warnings": warnings,
        }
        stage2_file = run_dir / "stage2.json"
        stage2_file.write_text(json.dumps(stage2_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return candidates, warnings
