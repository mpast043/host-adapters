#!/usr/bin/env python3
"""Tiered physics audit pipeline.

Tier A: internal theorem/math checks in capacity-demo.
Tier B: optional independent reruns in host-adapters.
Tier C: structured interpretation summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    name: str
    command: str
    cwd: str
    exit_code: int
    log_path: str


def run_and_log(name: str, command: list[str], cwd: Path, logs_dir: Path) -> CommandResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(command, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
    return CommandResult(
        name=name,
        command=" ".join(command),
        cwd=str(cwd),
        exit_code=proc.returncode,
        log_path=str(log_path),
    )


def choose_python(repo: Path) -> str:
    venv_py = repo / ".venv/bin/python"
    return str(venv_py) if venv_py.exists() else "python3"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiered physics audit")
    parser.add_argument("--host-repo", type=Path, default=Path("."))
    parser.add_argument(
        "--capacity-demo-repo",
        type=Path,
        default=Path("/Users/meganpastore/Clawdbot/Repos/capacity-demo"),
    )
    parser.add_argument("--run-tier-b", action="store_true", help="Execute Tier B independent reruns")
    parser.add_argument(
        "--tier-b-fast",
        action="store_true",
        help="Use reduced Tier B settings for quick matrix verification",
    )
    parser.add_argument("--output-json", type=Path, default=Path("docs/state/physics_audit_2026-02-27.json"))
    args = parser.parse_args()

    host_repo = args.host_repo.resolve()
    cap_repo = args.capacity_demo_repo.resolve()
    logs_dir = host_repo / "docs/state/physics_audit_logs"

    cap_py = choose_python(cap_repo)
    host_py = choose_python(host_repo)

    tier_a_cmds = [
        ("tier_a_test_framework_b", [cap_py, "-m", "pytest", "tests/test_framework_b.py", "-q"], cap_repo),
        (
            "tier_a_test_theorem_validation",
            [cap_py, "-m", "pytest", "tests/test_theorem_validation.py", "-q"],
            cap_repo,
        ),
        (
            "tier_a_validation_harness_quick",
            [cap_py, "scripts/run_framework_validation_b.py", "--quick"],
            cap_repo,
        ),
    ]

    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tier_a": [],
        "tier_b": [],
        "tier_c": {
            "bins": {
                "mathematically_proved_under_assumptions": [
                    "Monotone spectral filter definitions and boundedness",
                    "Capacity-only dependency and staircase obligations",
                    "Selection/evidence contract can be mechanically validated",
                ],
                "empirically_supported_local": [
                    "capacity-demo theorem and framework tests",
                    "quick validation harness outputs under fixed seeds",
                ],
                "conceptual_or_speculative": [
                    "prealgebraic fixed-point cosmology",
                    "global uniqueness conjectures",
                    "strong external physical interpretation claims without external benchmarks",
                ],
            }
        },
    }

    for name, cmd, cwd in tier_a_cmds:
        res = run_and_log(name, cmd, cwd, logs_dir)
        results["tier_a"].append(asdict(res))

    if args.run_tier_b:
        tier_b_cmds = []
        models = ["ising_open", "ising_cyclic", "heisenberg_open", "heisenberg_cyclic"]
        for L in (8, 16):
            a_size = "4" if L == 8 else "8"
            if args.tier_b_fast:
                chi_sweep = "2,4"
                fit_steps = "20"
                restarts = "1"
            else:
                chi_sweep = "2,4,8" if L == 8 else "2,4,8,16"
                fit_steps = "60" if L == 8 else "80"
                restarts = "2"
            for model in models:
                slug = model.replace("_", "-")
                tier_b_cmds.append(
                    (
                        f"tier_b_claim3p_{slug}_l{L}",
                        [
                            host_py,
                            "experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py",
                            "--L",
                            str(L),
                            "--A_size",
                            a_size,
                            "--model",
                            model,
                            "--chi_sweep",
                            chi_sweep,
                            "--restarts_per_chi",
                            restarts,
                            "--fit_steps",
                            fit_steps,
                            "--output",
                            f"outputs/physics_audit/{model}_L{L}",
                        ],
                        host_repo,
                    )
                )

        for name, cmd, cwd in tier_b_cmds:
            res = run_and_log(name, cmd, cwd, logs_dir)
            results["tier_b"].append(asdict(res))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output_json}")

    failures = [x for x in results["tier_a"] if x["exit_code"] != 0]
    if args.run_tier_b:
        failures.extend([x for x in results["tier_b"] if x["exit_code"] != 0])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
