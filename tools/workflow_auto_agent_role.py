#!/usr/bin/env python3
"""Single-role worker for agentic WORKFLOW_AUTO orchestration.

Roles:
- planner: generate next test plan from current run state
- researcher: gather external research signals and recommendations
- executor: run/resume WORKFLOW_AUTO and read resulting status
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def latest_run_dir(artifacts_root: Path) -> Path | None:
    runs = [p for p in artifacts_root.glob("RUN_*") if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def extract_run_dir(stdout: str, artifacts_root: Path) -> Path | None:
    for line in reversed([ln.strip() for ln in stdout.splitlines() if ln.strip()]):
        candidate = Path(line)
        if candidate.is_absolute() and candidate.is_dir() and candidate.name.startswith("RUN_"):
            return candidate
    return latest_run_dir(artifacts_root)


def run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    allowed_prefixes: list[list[str]],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if not any(cmd[: len(prefix)] == prefix for prefix in allowed_prefixes):
        raise RuntimeError(f"Role attempted non-allowlisted command: {' '.join(cmd)}")
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def role_planner(
    *,
    repo_root: Path,
    run_dir: Path,
    max_minutes: int,
    max_runs: int,
    cycle: int,
    allow_tier_c: bool,
    tier_c_justification: str,
    focus_objective: str,
) -> dict[str, Any]:
    status = load_json(run_dir / "results" / "workflow_auto_status.json")
    compute_target = str((status.get("mcp_targets") or {}).get("compute_target") or "")

    out_json = run_dir / "results" / "agentic" / f"planner_plan_cycle_{cycle:03d}.json"
    out_md = run_dir / "results" / "agentic" / f"planner_plan_cycle_{cycle:03d}.md"
    cmd = [
        "python3",
        "tools/plan_framework_selection_tests.py",
        "--repo-root",
        str(repo_root),
        "--artifacts-root",
        str(run_dir),
        "--catalog",
        "docs/physics/framework_selection_test_catalog_v1.json",
        "--max-minutes",
        str(max_minutes),
        "--max-runs",
        str(max_runs),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
        "--focus-objective",
        focus_objective,
    ]
    if compute_target.strip():
        cmd.extend(["--compute-target", compute_target.strip()])
    if allow_tier_c:
        cmd.extend(["--tier-c-justification", tier_c_justification.strip()])

    proc = run_checked(
        cmd,
        cwd=repo_root,
        allowed_prefixes=[["python3", "tools/plan_framework_selection_tests.py"]],
        env={**os.environ, "RUN_OVERRIDE_TIER_C": "1" if allow_tier_c else "0"},
    )
    plan = load_json(out_json)
    blocked = plan.get("blocked", [])
    tier_c_blocked = [x.get("test_id", "unknown") for x in blocked if str(x.get("tier", "")).upper() == "C"]
    selected = [x.get("execution_key", x.get("test_id", "")) for x in plan.get("selected", [])]

    return {
        "role": "planner",
        "ts_utc": utc_now(),
        "cycle": cycle,
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "selected_execution_keys": selected,
        "tier_c_blocked": tier_c_blocked,
        "plan_json": str(out_json),
        "plan_md": str(out_md),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def role_researcher(*, repo_root: Path, run_dir: Path, cycle: int, underdetermined_cycles: int) -> dict[str, Any]:
    out_json = run_dir / "results" / "agentic" / f"research_signal_cycle_{cycle:03d}.json"
    out_md = run_dir / "results" / "agentic" / f"research_signal_cycle_{cycle:03d}.md"
    cmd = [
        "python3",
        "tools/research_framework_selection.py",
        "--run-dir",
        str(run_dir),
        "--underdetermined-cycles",
        str(underdetermined_cycles),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = run_checked(
        cmd,
        cwd=repo_root,
        allowed_prefixes=[["python3", "tools/research_framework_selection.py"]],
    )
    payload = load_json(out_json)
    rec = payload.get("recommendations", {})
    return {
        "role": "researcher",
        "ts_utc": utc_now(),
        "cycle": cycle,
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "escalate_tier_c": bool(rec.get("escalate_tier_c", False)),
        "recommended_execution_keys": rec.get("recommended_execution_keys", []),
        "signal_score": rec.get("signal_score", 0),
        "research_json": str(out_json),
        "research_md": str(out_md),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def role_executor(
    *,
    repo_root: Path,
    artifacts_root: Path,
    cycle: int,
    seed: int,
    allow_tier_c: bool,
    tier_c_justification: str,
    focus_objective: str,
) -> dict[str, Any]:
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
        "--focus-objective",
        focus_objective,
    ]
    if allow_tier_c:
        cmd.extend(["--tier-c-justification", tier_c_justification.strip()])
    env = os.environ.copy()
    if allow_tier_c:
        env["RUN_OVERRIDE_TIER_C"] = "1"
    proc = run_checked(
        cmd,
        cwd=repo_root,
        allowed_prefixes=[["python3", "tools/run_workflow_auto.py"]],
        env=env,
    )
    run_dir = extract_run_dir(proc.stdout, artifacts_root)
    status = {}
    verdict = {}
    if run_dir is not None:
        status = load_json(run_dir / "results" / "workflow_auto_status.json")
        verdict = load_json(run_dir / "results" / "VERDICT.json")
    return {
        "role": "executor",
        "ts_utc": utc_now(),
        "cycle": cycle,
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "run_dir": str(run_dir) if run_dir else None,
        "mode": str(status.get("mode", "UNKNOWN")).upper(),
        "selection_status": str(status.get("selection_status", "UNKNOWN")).upper(),
        "overall_status": str(verdict.get("overall_status", "UNKNOWN")).upper(),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one agent role for WORKFLOW_AUTO")
    parser.add_argument("--role", required=True, choices=["planner", "researcher", "executor"])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--max-runs", type=int, default=80)
    parser.add_argument(
        "--focus-objective",
        type=str,
        default="ALL",
        choices=["ALL", "A", "B", "C"],
    )
    parser.add_argument("--allow-tier-c", action="store_true")
    parser.add_argument(
        "--tier-c-justification",
        type=str,
        default="Escalate Tier C to resolve persistent UNDERDETERMINED selection.",
    )
    parser.add_argument("--underdetermined-cycles", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir(artifacts_root)

    result: dict[str, Any]
    if args.role == "executor":
        result = role_executor(
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            cycle=args.cycle,
            seed=args.seed,
            allow_tier_c=args.allow_tier_c,
            tier_c_justification=args.tier_c_justification,
            focus_objective=args.focus_objective,
        )
    else:
        if run_dir is None:
            result = {
                "role": args.role,
                "cycle": args.cycle,
                "ok": False,
                "exit_code": 1,
                "error": "No RUN_* directory available for planner/researcher role",
            }
        elif args.role == "planner":
            result = role_planner(
                repo_root=repo_root,
                run_dir=run_dir,
                max_minutes=args.max_minutes,
                max_runs=args.max_runs,
                cycle=args.cycle,
                allow_tier_c=args.allow_tier_c,
                tier_c_justification=args.tier_c_justification,
                focus_objective=args.focus_objective,
            )
        else:
            result = role_researcher(
                repo_root=repo_root,
                run_dir=run_dir,
                cycle=args.cycle,
                underdetermined_cycles=args.underdetermined_cycles,
            )

    write_json(args.output_json.resolve(), result)
    print(str(args.output_json.resolve()))
    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
