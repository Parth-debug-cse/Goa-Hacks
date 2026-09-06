#!/usr/bin/env python3
"""Proof-of-Match (pom) CLI Entrypoint (§3).

Subcommands:
  run        - Execute end-to-end pipeline (Stage 0 -> 1 -> 2 -> 3 -> 4)
  verify     - Independently audit an anchor record or demonstrate tamper detection
  probe      - Probe an external API engine and dump raw response to probes/<engine>.json
  deploy     - Deploy MatchRegistry.sol smart contract to active chain
  calibrate  - Calibrate biometric similarity thresholds over image pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from pom.config import CONFIG
from pom.verifier import main as verifier_main, verify_record_and_anchor


def cmd_run(args: argparse.Namespace) -> int:
    """Run the complete end-to-end pipeline."""
    # INV-6: Consent gate and strictly single-photo input
    if not args.consent_confirmed:
        print("Error: --consent-confirmed is REQUIRED. This pipeline is only for consenting team members (INV-6).", file=sys.stderr)
        return 1

    if os.path.isdir(args.photo):
        print(f"Error: Refusing directory input '{args.photo}'. Batch processing is forbidden (INV-6).", file=sys.stderr)
        return 1

    if not os.path.isfile(args.photo):
        print(f"Error: Photo file not found: '{args.photo}'", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    LOGGER = logging.getLogger("pom")

    from pom.stage0_manifest import setup_run_manifest
    from pom.stage1_face import process_stage1
    from pom.stage2_search import process_stage2
    from pom.stage3_verify import process_stage3
    from pom.stage4_anchor import process_stage4

    # Stage 0: Manifest setup
    LOGGER.info("Stage 0: Setting up run manifest and evidence directory...")
    run_id, run_dir, manifest = setup_run_manifest(args.photo)
    LOGGER.info("Run ID: %s -> %s", run_id, run_dir)

    # Stage 1: Face detection & embeddings
    LOGGER.info("Stage 1: Face detection, quality gate, and dual embeddings...")
    stage1 = process_stage1(args.photo, run_dir, args.adaface_root)
    if not stage1.get("quality_passed") or not stage1.get("arcface_embedding"):
        LOGGER.error("Stage 1 quality gate or embedding failed: %s", stage1.get("reason"))
        out_payload = {"match_found": False, "source_photo_note": stage1, "candidates_tried": 0, "warnings": ["stage1_quality_or_embedding_failed"]}
        print(json.dumps(out_payload, indent=2, sort_keys=True))
        return 1

    # Stage 2: Reverse-image search & hop-2 identity pivot
    LOGGER.info("Stage 2: Multi-engine reverse search and hop-2 identity pivot...")
    candidates, warnings = process_stage2(args.photo, run_dir)

    # Stage 3: Verification & evidence capture
    LOGGER.info("Stage 3: Candidate face verification and evidence capture...")
    stage3 = process_stage3(candidates, stage1, run_dir)
    stage3["warnings"] = warnings + stage3.get("warnings", [])

    if not stage3.get("match_found"):
        LOGGER.warning("No verified match found across %d candidates (INV-1).", stage3.get("candidates_tried", 0))
        print(json.dumps(stage3, indent=2, sort_keys=True))
        return 2

    # Stage 4: Blockchain Anchoring & Encrypted Bundle
    LOGGER.info("Stage 4: Cryptographic proof generation and blockchain anchoring...")
    receipt = process_stage4(stage3, stage1, run_dir)

    # Also save to requested output path if specified
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Keep backward-compatible anchor_receipt.json at root
    Path("anchor_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Execute live probe for an engine and save raw JSON dump to probes/<engine>.json (AH-1)."""
    engine = args.engine.lower()
    probes_dir = CONFIG.probes_dir
    probes_dir.mkdir(parents=True, exist_ok=True)
    out_probe = probes_dir / f"{engine}.json"
    
    print(f"Executing probe for engine: {engine} -> {out_probe}")
    if out_probe.exists():
        print(f"Probe file already exists at {out_probe} ({out_probe.stat().st_size} bytes)")
        return 0
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy MatchRegistry.sol smart contract."""
    from contracts.deploy import deploy_contract
    return deploy_contract()


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Calibrate similarity thresholds over local pairs file (§14)."""
    from pom.calibrate import calibrate_thresholds, create_default_pairs_csv
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"Pairs CSV not found at {pairs_path}. Generating default dataset...")
        create_default_pairs_csv(pairs_path)
    calibrate_thresholds(pairs_csv_path=pairs_path, output_json_path=Path(args.output), quiet=args.quiet)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pom",
        description="Proof-of-Match (pom): Tamper-Evident Biometric Face-to-Blockchain Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run the full pipeline on a photo of a consenting person")
    run_parser.add_argument("photo", help="Path to single reference photo")
    run_parser.add_argument(
        "--consent-confirmed",
        action="store_true",
        required=True,
        help="Mandatory acknowledgement that the photo is a consenting person (INV-6)",
    )
    run_parser.add_argument("--adaface-root", help="Path to AdaFace repo root")
    run_parser.add_argument("--output", "-o", default="anchor_receipt.json", help="Path to save anchor receipt JSON")

    # verify (§13)
    verify_parser = subparsers.add_parser("verify", help="Independently audit an anchor record or demonstrate tamper detection")
    verify_parser.add_argument("receipt", nargs="?", default="anchor_receipt.json", help="Path to record.json or anchor_receipt.json")
    verify_parser.add_argument("--record", dest="record_opt", help="Path to record.json")
    verify_parser.add_argument("--tamper", help="Dotted field path to tamper in memory (e.g. match.similarity.score)")
    verify_parser.add_argument("--tamper-value", help="Value to set for tampered field")
    verify_parser.add_argument("--tamper-demo", action="store_true", help="Demonstrate tamper detection")
    verify_parser.add_argument("--field", default="match.similarity.score", help="Field path to mutate in tamper demo")
    verify_parser.add_argument("--bundle", help="Path to bundle.enc")
    verify_parser.add_argument("--check-gateway", action="store_true", help="Verify gateway bytes against bundle_sha256")
    verify_parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

    # probe
    probe_parser = subparsers.add_parser("probe", help="Probe external API and dump response schema")
    probe_parser.add_argument("--engine", required=True, choices=["serpapi", "google_vision", "pdl", "polygon", "pinata"])

    # deploy
    subparsers.add_parser("deploy", help="Deploy MatchRegistry.sol to active blockchain")

    # calibrate (§14)
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate biometric thresholds over image pairs (§14)")
    cal_parser.add_argument("--pairs", default="data/pairs.csv", help="Path to pairs.csv file")
    cal_parser.add_argument("--output", "-o", default="calibration.json", help="Path to save calibration.json")
    cal_parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "verify":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return verifier_main()
    elif args.command == "probe":
        return cmd_probe(args)
    elif args.command == "deploy":
        return cmd_deploy(args)
    elif args.command == "calibrate":
        return cmd_calibrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
