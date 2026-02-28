#!/usr/bin/env python3
"""Autonomous supervisor for WORKFLOW_AUTO.

Runs `run_workflow_auto.py` in resume mode across multiple cycles so a PARTIAL
run can continue without manual relaunch. Stops on COMPLETE, STOPPED, or when
max cycles are reached.
"""

from __future__ import annotations

import argparse
import json
import os
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


def run_once(
    repo_root: Path,
    artifacts_root: Path,
    seed: int,
    *,
    enable_tier_c: bool,
    tier_c_justification: str,
) -> tuple[int, Path | None]:
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
    if enable_tier_c and tier_c_justification.strip():
        cmd.extend(["--tier-c-justification", tier_c_justification.strip()])

    env = os.environ.copy()
    if enable_tier_c:
        env["RUN_OVERRIDE_TIER_C"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )  # noqa: S603
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


def read_plan_state(run_dir: Path) -> tuple[bool, list[str]]:
    proposal = _load_json(run_dir / "results" / "science" / "exploration_proposals.json")
    blocked = proposal.get("blocked", [])
    tier_c_blocked = [
        str(item.get("test_id", "unknown"))
        for item in blocked
        if str(item.get("tier", "")).upper() == "C"
    ]
    stop_triggered = bool((proposal.get("stop_condition") or {}).get("triggered", False))
    return stop_triggered, tier_c_blocked


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
    parser.add_argument(
        "--until-resolved",
        action="store_true",
        help="Keep cycling until run resolves to COMPLETE or STOPPED (ignores --max-cycles).",
    )
    parser.add_argument(
        "--tier-c-after-cycle",
        type=int,
        default=0,
        help="Enable Tier C override starting at this cycle number (0 disables).",
    )
    parser.add_argument(
        "--tier-c-justification",
        type=str,
        default="Escalate Tier C to resolve persistent UNDERDETERMINED selection.",
        help="One-line justification used when Tier C override is enabled.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()

    seed = args.seed
    cycle = 1
    while True:
        tier_c_enabled = bool(args.tier_c_after_cycle) and cycle >= args.tier_c_after_cycle
        cycle_label = f"{cycle}/∞" if args.until_resolved else f"{cycle}/{args.max_cycles}"
        print(
            f"[workflow-auto-supervisor] cycle={cycle_label} seed={seed} "
            f"tier_c_override={'on' if tier_c_enabled else 'off'}"
        )
        rc, run_dir = run_once(
            repo_root,
            artifacts_root,
            seed,
            enable_tier_c=tier_c_enabled,
            tier_c_justification=args.tier_c_justification,
        )
        if run_dir is None:
            print("[workflow-auto-supervisor] ERROR: no RUN_* directory found after execution")
            return 1

        overall, mode, selection = read_status(run_dir)
        stop_triggered, tier_c_blocked = read_plan_state(run_dir)
        print(
            "[workflow-auto-supervisor] "
            f"run={run_dir.name} overall={overall} mode={mode} selection={selection} rc={rc}"
        )

        if mode == "STOPPED" or overall == "STOPPED":
            return 1
        if overall == "COMPLETE":
            return 0

        # Continue automatically when the run is still underdetermined.
        if selection == "UNDERDETERMINED":
            if args.until_resolved:
                seed += 1
                cycle += 1
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
                continue
            if cycle < args.max_cycles:
                seed += 1
                cycle += 1
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
                continue

            reason = (
                "UNDERDETERMINED persisted through max cycles; no conclusive selection outcome."
            )
            if stop_triggered and tier_c_blocked:
                reason += (
                    " Tier C queue remained blocked: "
                    + ", ".join(tier_c_blocked)
                    + ". Use --tier-c-after-cycle with justification to escalate."
                )
            print(f"[workflow-auto-supervisor] {reason}")
            return 2

        # PARTIAL with non-undertermined outcome, or max cycles hit.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
