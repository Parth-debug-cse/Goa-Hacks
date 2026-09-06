"""Stage 0: Manifest setup, run_id generation, and evidence directory creation (§3)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pom.config import CONFIG
from pom.provenance import set_provenance_log_path


def _get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def setup_run_manifest(photo_path: str, out_root: Path | None = None) -> tuple[str, Path, dict[str, Any]]:
    """Initialize a new run directory out/<run_id>/ with manifest.json and evidence/."""
    run_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    photo_file = Path(photo_path)
    photo_bytes = photo_file.read_bytes()
    photo_hash = "0x" + hashlib.sha256(photo_bytes).hexdigest()

    # Generate unique run_id e.g. 20260906_143000_a1b2
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{photo_hash[2:8]}"
    
    root = out_root or CONFIG.out_dir
    run_dir = root / run_id
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Copy input photo to evidence
    input_evidence_path = evidence_dir / "input.jpg"
    input_evidence_path.write_bytes(photo_bytes)

    # Point provenance logger to out/<run_id>/evidence/requests.jsonl
    set_provenance_log_path(evidence_dir / "requests.jsonl")

    manifest = {
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp,
        "git_commit": _get_git_commit(),
        "input_photo": str(photo_path),
        "input_photo_sha256": photo_hash,
        "input_photo_size_bytes": len(photo_bytes),
    }

    manifest_file = run_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return run_id, run_dir, manifest
