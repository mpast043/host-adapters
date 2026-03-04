#!/usr/bin/env python3
"""
Ledger Generator - Mechanically Generated Single Source of Truth

This script scans all run artifacts and generates a comprehensive ledger
with test name, runner version, run dir, key metrics, verdict, and links.

Usage:
    python3 ledger_generator.py --run-dir /path/to/runs --output ledger.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_run_metadata(run_dir: Path) -> Optional[Dict]:
    """Extract metadata from a run directory."""
    metadata_files = list(run_dir.glob("metadata.json"))
    if not metadata_files:
        return None

    with open(metadata_files[0]) as f:
        metadata = json.load(f)

    return metadata


def extract_verdict(run_dir: Path) -> Optional[Dict]:
    """Extract verdict from a run directory."""
    verdict_files = list(run_dir.glob("verdict.json"))
    if not verdict_files:
        return None

    with open(verdict_files[0]) as f:
        verdict = json.load(f)

    return verdict


def extract_fits(run_dir: Path) -> Optional[Dict]:
    """Extract fits from a run directory."""
    fits_files = list(run_dir.glob("fits.json"))
    if not fits_files:
        return None

    with open(fits_files[0]) as f:
        fits = json.load(f)

    return fits


def extract_summary(run_dir: Path) -> Dict[str, Any]:
    """Extract summary metrics from a run directory."""
    summary = {}

    # Get metadata
    metadata = extract_run_metadata(run_dir)
    if metadata:
        summary["test"] = metadata.get("test", "unknown")
        summary["version"] = metadata.get("version", "unknown")
        summary["timestamp"] = metadata.get("timestamp", "")
        summary["run_id"] = metadata.get("run_id", "")

    # Get verdict
    verdict = extract_verdict(run_dir)
    if verdict:
        summary["verdict"] = verdict.get("verdict", "UNKNOWN")
        summary["status"] = verdict.get("status", "unknown")
        summary["passed"] = verdict.get("passed", {})
        summary["delta_aic"] = verdict.get("delta_aic", 0)

    # Get fits
    fits = extract_fits(run_dir)
    if fits:
        summary["delta_aic"] = fits.get("delta_aic", 0)
        summary["delta_bic"] = fits.get("delta_bic", 0)
        summary["saturating"] = fits.get("saturating", {})
        summary["loglinear"] = fits.get("loglinear", {})

    return summary


def generate_ledger(run_dir: Path, output_path: Path) -> Dict:
    """Generate ledger from run directories."""
    ledger = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_dir": str(run_dir),
        "runs": []
    }

    # Find all run directories
    run_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])

    for run_dir_item in run_dirs:
        summary = extract_summary(run_dir_item)
        if summary:
            ledger["runs"].append({
                "run_dir": str(run_dir_item),
                "summary": summary
            })

    # Compute aggregate statistics
    ledger["aggregate"] = compute_aggregates(ledger["runs"])

    return ledger


def compute_aggregates(runs: List[Dict]) -> Dict:
    """Compute aggregate statistics from runs."""
    aggregates = {
        "total_runs": len(runs),
        "verdict_counts": {},
        "tests_seen": set(),
        "latest_run": None
    }

    for run in runs:
        summary = run.get("summary", {})
        verdict = summary.get("verdict", "UNKNOWN")

        aggregates["verdict_counts"][verdict] = aggregates["verdict_counts"].get(verdict, 0) + 1
        aggregates["tests_seen"].add(summary.get("test", "unknown"))

        if summary.get("timestamp"):
            if aggregates["latest_run"] is None or summary["timestamp"] > aggregates["latest_run"].get("timestamp", ""):
                aggregates["latest_run"] = summary

    aggregates["tests_seen"] = sorted(list(aggregates["tests_seen"]))

    return aggregates


def main():
    p = argparse.ArgumentParser(description="Generate auditable ledger from run artifacts")
    p.add_argument("--run-dir", required=True, help="Directory containing run artifacts")
    p.add_argument("--output", required=True, help="Output ledger JSON file")
    a = p.parse_args()

    run_dir = Path(a.run_dir)
    output_path = Path(a.output)

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        sys.exit(1)

    print(f"[Ledger] Generating ledger from {run_dir}")

    ledger = generate_ledger(run_dir, output_path)

    # Write ledger
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ledger, f, indent=2, default=str)

    print(f"[Ledger] Ledger written to {output_path}")
    print(f"[Ledger] Total runs: {ledger['aggregate']['total_runs']}")
    print(f"[Ledger] Verdicts: {ledger['aggregate']['verdict_counts']}")
    print(f"[Ledger] Tests: {ledger['aggregate']['tests_seen']}")


if __name__ == "__main__":
    main()
