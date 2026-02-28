#!/usr/bin/env python3
"""Autonomous supervisor for WORKFLOW_AUTO.

Runs `run_workflow_auto.py` in resume mode across multiple cycles so a PARTIAL
run can continue without manual relaunch. Stops on COMPLETE, STOPPED, or when
max cycles are reached.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _latest_run_dir(artifacts_root: Path) -> Path | None:
    runs = [p for p in artifacts_root.glob("RUN_*") if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _extract_run_dir(stdout: str, artifacts_root: Path) -> Path | None:
    for line in reversed([ln.strip() for ln in stdout.splitlines() if ln.strip()]):
        p = Path(line)
        if p.is_absolute() and p.exists() and p.is_dir() and p.name.startswith("RUN_"):
            return p
    return _latest_run_dir(artifacts_root)


def run_once(repo_root: Path, artifacts_root: Path, seed: int) -> tuple[int, Path | None]:
    cmd = [
        "python3",
        "tools/run_workflow_auto.py",
        "--repo-root",
        str(repo_root),
        "--artifacts-root",
        str(artifacts_root),
        "--resume-latest",
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True, check=False)  # noqa: S603
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    return proc.returncode, _extract_run_dir(proc.stdout, artifacts_root)


def read_status(run_dir: Path) -> tuple[str, str, str]:
    status = _load_json(run_dir / "results" / "workflow_auto_status.json")
    verdict = _load_json(run_dir / "results" / "VERDICT.json")
    overall = str(verdict.get("overall_status", "UNKNOWN")).upper()
    mode = str(status.get("mode", "UNKNOWN")).upper()
    selection = str(status.get("selection_status", "UNKNOWN")).upper()
    return overall, mode, selection


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous WORKFLOW_AUTO supervisor")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cycles", type=int, default=6)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()

    seed = args.seed
    for cycle in range(1, args.max_cycles + 1):
        print(f"[workflow-auto-supervisor] cycle={cycle}/{args.max_cycles} seed={seed}")
        rc, run_dir = run_once(repo_root, artifacts_root, seed)
        if run_dir is None:
            print("[workflow-auto-supervisor] ERROR: no RUN_* directory found after execution")
            return 1

        overall, mode, selection = read_status(run_dir)
        print(
            "[workflow-auto-supervisor] "
            f"run={run_dir.name} overall={overall} mode={mode} selection={selection} rc={rc}"
        )

        if mode == "STOPPED" or overall == "STOPPED":
            return 1
        if overall == "COMPLETE":
            return 0

        # Continue automatically when the run is still underdetermined.
        if selection == "UNDERDETERMINED" and cycle < args.max_cycles:
            seed += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            continue

        # PARTIAL with non-undertermined outcome, or max cycles hit.
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

