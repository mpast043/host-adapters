#!/usr/bin/env python3
"""Multi-agent WORKFLOW_AUTO coordinator.

Spawns scoped subprocess workers:
- planner agent
- researcher agent
- executor agent

The coordinator owns policy/stop rules and writes an agentic event ledger.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def latest_run_dir(artifacts_root: Path) -> Path | None:
    runs = [p for p in artifacts_root.glob("RUN_*") if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def run_role(
    *,
    role: str,
    repo_root: Path,
    artifacts_root: Path,
    run_dir: Path | None,
    cycle: int,
    seed: int,
    max_minutes: int,
    max_runs: int,
    allow_tier_c: bool,
    tier_c_justification: str,
    underdetermined_cycles: int,
    agentic_dir: Path,
) -> dict[str, Any]:
    output_json = agentic_dir / f"{role}_result_cycle_{cycle:03d}.json"
    cmd = [
        "python3",
        "tools/workflow_auto_agent_role.py",
        "--role",
        role,
        "--repo-root",
        str(repo_root),
        "--artifacts-root",
        str(artifacts_root),
        "--cycle",
        str(cycle),
        "--seed",
        str(seed),
        "--max-minutes",
        str(max_minutes),
        "--max-runs",
        str(max_runs),
        "--tier-c-justification",
        tier_c_justification.strip(),
        "--underdetermined-cycles",
        str(underdetermined_cycles),
        "--output-json",
        str(output_json),
    ]
    if run_dir is not None:
        cmd.extend(["--run-dir", str(run_dir)])
    if allow_tier_c:
        cmd.append("--allow-tier-c")

    started = utc_now()
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True, check=False)  # noqa: S603
    finished = utc_now()
    result = read_json(output_json)
    return {
        "role": role,
        "cycle": cycle,
        "seed": seed,
        "allow_tier_c": allow_tier_c,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "worker_exit_code": proc.returncode,
        "worker_stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "worker_stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent coordinator for WORKFLOW_AUTO")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cycles", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--max-runs", type=int, default=80)
    parser.add_argument("--until-resolved", action="store_true")
    parser.add_argument("--tier-c-after-cycle", type=int, default=0)
    parser.add_argument(
        "--tier-c-justification",
        type=str,
        default="Escalate Tier C to resolve persistent UNDERDETERMINED selection.",
    )
    parser.add_argument("--research-on-underdetermined", action="store_true")
    parser.add_argument("--research-auto-escalate-tier-c", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()

    # Bootstrap run if needed.
    run_dir = latest_run_dir(artifacts_root)
    if run_dir is None:
        bootstrap_dir = artifacts_root / "agentic_bootstrap"
        bootstrap = run_role(
            role="executor",
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            run_dir=None,
            cycle=0,
            seed=args.seed,
            max_minutes=args.max_minutes,
            max_runs=args.max_runs,
            allow_tier_c=False,
            tier_c_justification=args.tier_c_justification,
            underdetermined_cycles=0,
            agentic_dir=bootstrap_dir,
        )
        run_dir_value = ((bootstrap.get("result") or {}).get("run_dir") or "").strip()
        run_dir = Path(run_dir_value) if run_dir_value else latest_run_dir(artifacts_root)
        if run_dir is None:
            print("[workflow-auto-agentic] ERROR: could not initialize RUN_* directory")
            return 1

    agentic_dir = run_dir / "results" / "agentic"
    events_path = run_dir / "logs" / "agentic_events.jsonl"

    underdetermined_streak = 0
    dynamic_tier_c = False
    seed = args.seed
    cycle = 1

    while True:
        if not args.until_resolved and cycle > args.max_cycles:
            summary = {
                "ts_utc": utc_now(),
                "event": "max_cycles_reached",
                "cycle": cycle - 1,
                "status": "UNRESOLVED",
            }
            append_event(events_path, summary)
            print("[workflow-auto-agentic] unresolved at max cycles")
            return 2

        tier_c_enabled = dynamic_tier_c or (bool(args.tier_c_after_cycle) and cycle >= args.tier_c_after_cycle)
        cycle_label = f"{cycle}/∞" if args.until_resolved else f"{cycle}/{args.max_cycles}"
        print(
            f"[workflow-auto-agentic] cycle={cycle_label} seed={seed} "
            f"tier_c_override={'on' if tier_c_enabled else 'off'}"
        )

        planner_event = run_role(
            role="planner",
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            run_dir=run_dir,
            cycle=cycle,
            seed=seed,
            max_minutes=args.max_minutes,
            max_runs=args.max_runs,
            allow_tier_c=tier_c_enabled,
            tier_c_justification=args.tier_c_justification,
            underdetermined_cycles=underdetermined_streak,
            agentic_dir=agentic_dir,
        )
        append_event(events_path, {"ts_utc": utc_now(), "event": "planner_finished", **planner_event})

        if planner_event["worker_exit_code"] != 0:
            print("[workflow-auto-agentic] planner failed")
            return 1

        research_event = None
        if args.research_on_underdetermined:
            research_event = run_role(
                role="researcher",
                repo_root=repo_root,
                artifacts_root=artifacts_root,
                run_dir=run_dir,
                cycle=cycle,
                seed=seed,
                max_minutes=args.max_minutes,
                max_runs=args.max_runs,
                allow_tier_c=tier_c_enabled,
                tier_c_justification=args.tier_c_justification,
                underdetermined_cycles=max(1, underdetermined_streak),
                agentic_dir=agentic_dir,
            )
            append_event(events_path, {"ts_utc": utc_now(), "event": "researcher_finished", **research_event})
            if research_event["worker_exit_code"] != 0:
                print("[workflow-auto-agentic] researcher failed")
                return 1

            escalate = bool((research_event.get("result") or {}).get("escalate_tier_c", False))
            if args.research_auto_escalate_tier_c and escalate and not tier_c_enabled:
                dynamic_tier_c = True
                append_event(
                    events_path,
                    {
                        "ts_utc": utc_now(),
                        "event": "tier_c_escalated_from_research",
                        "cycle": cycle,
                    },
                )

        executor_event = run_role(
            role="executor",
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            run_dir=run_dir,
            cycle=cycle,
            seed=seed,
            max_minutes=args.max_minutes,
            max_runs=args.max_runs,
            allow_tier_c=dynamic_tier_c or tier_c_enabled,
            tier_c_justification=args.tier_c_justification,
            underdetermined_cycles=underdetermined_streak,
            agentic_dir=agentic_dir,
        )
        append_event(events_path, {"ts_utc": utc_now(), "event": "executor_finished", **executor_event})

        if executor_event["worker_exit_code"] != 0:
            print("[workflow-auto-agentic] executor failed")
            return 1

        result = executor_event.get("result") or {}
        run_dir_value = (result.get("run_dir") or "").strip()
        if run_dir_value:
            run_dir = Path(run_dir_value)
        mode = str(result.get("mode", "UNKNOWN")).upper()
        overall = str(result.get("overall_status", "UNKNOWN")).upper()
        selection = str(result.get("selection_status", "UNKNOWN")).upper()

        print(
            "[workflow-auto-agentic] "
            f"run={run_dir.name} overall={overall} mode={mode} selection={selection}"
        )

        if mode == "STOPPED" or overall == "STOPPED":
            return 1
        if overall == "COMPLETE":
            return 0

        if selection == "UNDERDETERMINED":
            underdetermined_streak += 1
        else:
            underdetermined_streak = 0

        seed += 1
        cycle += 1
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

