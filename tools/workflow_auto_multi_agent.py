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
import csv
import datetime as dt
import hashlib
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


def load_claim_map(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def claim_gate_status(path: Path) -> dict[str, Any]:
    data = load_claim_map(path)
    claims = data.get("claims", [])
    unresolved_statuses = {"DOC_ONLY", "PARTIAL", "OPEN"}
    unresolved = []
    for claim in claims:
        status = str(claim.get("status", "")).upper().strip()
        if status in unresolved_statuses:
            unresolved.append(str(claim.get("id", "unknown")))
    unresolved_sorted = sorted(set(unresolved))
    digest = hashlib.sha256(",".join(unresolved_sorted).encode("utf-8")).hexdigest() if unresolved_sorted else ""
    return {
        "enabled": path.exists(),
        "path": str(path),
        "total_claims": len(claims),
        "unresolved_count": len(unresolved_sorted),
        "unresolved_claim_ids": unresolved_sorted,
        "unresolved_digest": digest,
    }


def latest_campaign_by_test(run_dir: Path) -> dict[str, dict[str, str]]:
    campaign_csv = run_dir / "results" / "science" / "campaign" / "campaign_index.csv"
    if not campaign_csv.exists():
        return {}
    latest: dict[str, dict[str, str]] = {}
    with campaign_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            test_id = str(row.get("test_id", "")).strip()
            if test_id:
                latest[test_id] = row
    return latest


def refresh_claim_map_live(base_claim_map_path: Path, run_dir: Path) -> Path:
    """Create/update a run-local live claim map from baseline map + run evidence."""
    base = load_claim_map(base_claim_map_path)
    if not base:
        return base_claim_map_path

    claim_dir = run_dir / "results" / "claim_map"
    claim_dir.mkdir(parents=True, exist_ok=True)
    live_path = claim_dir / "framework_pdf_claim_map_live.json"

    data = json.loads(json.dumps(base))  # deep copy
    by_test = latest_campaign_by_test(run_dir)

    # Conservative auto-promotion only for explicitly mapped partial claims.
    evidence_map = {
        "W01": "claim2_seed_perturbation",
        "W05": "claim3_optionb_regime_check",
        "W02": "claim_w02_poset_infimum",
        "W06": "claim_w06_depth_vector_monotonicity",
        "W08": "claim_w08_class_splitting_monotonicity",
        "W13": "claim_w13_cobs_decomposition_compat",
        "W14": "claim_w14_ejection_expands_core",
        "W16": "claim_w16_time_consistency_monotone",
    }

    for claim in data.get("claims", []):
        cid = str(claim.get("id", "")).strip()
        test_id = evidence_map.get(cid)
        if not test_id:
            continue
        row = by_test.get(test_id)
        if row is None:
            continue
        verdict = str(row.get("verdict", "")).upper().strip()
        if verdict in {"PASS", "SUPPORTED", "ACCEPTED"}:
            claim["status"] = "LOCAL_EXEC"
            evidence_paths = claim.get("evidence_paths", [])
            if not isinstance(evidence_paths, list):
                evidence_paths = []
            campaign_path = str((run_dir / "results" / "science" / "campaign" / "campaign_index.csv"))
            if campaign_path not in evidence_paths:
                evidence_paths.append(campaign_path)
            claim["evidence_paths"] = evidence_paths

    data["live_updated_utc"] = utc_now()
    data["live_source_run_dir"] = str(run_dir)
    live_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return live_path


def runnable_claim_actions(unresolved_claim_ids: list[str]) -> list[str]:
    # Current executable coverage in workflow catalog.
    mapping = {
        "W01": "claim2_seed_perturbation",
        "W05": "claim3_optionb_regime_check",
        "W02": "claim_w02_poset_infimum",
        "W06": "claim_w06_depth_vector_monotonicity",
        "W08": "claim_w08_class_splitting_monotonicity",
        "W13": "claim_w13_cobs_decomposition_compat",
        "W14": "claim_w14_ejection_expands_core",
        "W16": "claim_w16_time_consistency_monotone",
    }
    out: list[str] = []
    for cid in unresolved_claim_ids:
        key = mapping.get(cid)
        if key and key not in out:
            out.append(key)
    return out


def write_live_brief(
    *,
    path: Path,
    history_path: Path,
    run_dir: Path,
    cycle: int,
    cycle_label: str,
    seed: int,
    focus_objective: str,
    tier_c_override: bool,
    underdetermined_streak: int,
    planner_event: dict[str, Any],
    research_event: dict[str, Any] | None,
    executor_event: dict[str, Any] | None,
    claim_gate: dict[str, Any] | None = None,
) -> None:
    planner_result = planner_event.get("result") or {}
    research_result = (research_event or {}).get("result") or {}
    executor_result = (executor_event or {}).get("result") or {}

    selected = planner_result.get("selected_execution_keys") or []
    recommended = research_result.get("recommended_execution_keys") or []
    mode = str(executor_result.get("mode", "UNKNOWN")).upper()
    overall = str(executor_result.get("overall_status", "UNKNOWN")).upper()
    selection = str(executor_result.get("selection_status", "UNKNOWN")).upper()
    executed_tests: list[str] = []
    campaign_csv = run_dir / "results" / "science" / "campaign" / "campaign_index.csv"
    if campaign_csv.exists():
        try:
            with campaign_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    tid = str(row.get("test_id", "")).strip()
                    if tid and tid not in executed_tests:
                        executed_tests.append(tid)
        except Exception:  # noqa: BLE001
            executed_tests = []

    lines = [
        "# Workflow Physics Live Brief",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Run: `{run_dir.name}`",
        f"- Cycle: `{cycle_label}` (seed `{seed}`)",
        f"- Focus objective: `{focus_objective}`",
        f"- Tier C override: `{'on' if tier_c_override else 'off'}`",
        f"- Underdetermined streak: `{underdetermined_streak}`",
        "",
        "## Role Status",
        f"- Planner exit: `{planner_event.get('worker_exit_code')}`",
        f"- Researcher exit: `{(research_event or {}).get('worker_exit_code', 'not-run')}`",
        f"- Executor exit: `{(executor_event or {}).get('worker_exit_code', 'not-run')}`",
        "",
        "## Current Outcome",
        f"- Overall: `{overall}`",
        f"- Mode: `{mode}`",
        f"- Selection: `{selection}`",
        "",
        "## Planned Execution Keys",
    ]
    if selected:
        lines.extend([f"- `{key}`" for key in selected])
    else:
        lines.append("- none")

    lines.extend(["", "## Research Recommendations"])
    if recommended:
        lines.extend([f"- `{key}`" for key in recommended])
    else:
        lines.append("- none")

    lines.extend(["", "## Executed Tests (Current Run)"])
    if executed_tests:
        lines.extend([f"- `{key}`" for key in executed_tests])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Artifact Pointers",
            f"- `{run_dir}/results/workflow_auto_status_latest.json`",
            f"- `{run_dir}/results/VERDICT_latest.json`",
            f"- `{run_dir}/results/science/campaign/campaign_report.md`",
            f"- `{run_dir}/results/selection/selection_report.md`",
            f"- `{run_dir}/logs/agentic_events.jsonl`",
        ]
    )
    if claim_gate is not None:
        lines.extend(
            [
                "",
                "## Claim Map Gate",
                f"- Enabled: `{claim_gate.get('enabled', False)}`",
                f"- Unresolved count: `{claim_gate.get('unresolved_count', 0)}`",
                f"- Claim map path: `{claim_gate.get('path', '')}`",
            ]
        )
        unresolved = claim_gate.get("unresolved_claim_ids", []) or []
        if unresolved:
            lines.append("- Unresolved IDs: " + ", ".join(f"`{x}`" for x in unresolved[:20]))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    history_row = {
        "ts_utc": utc_now(),
        "cycle": cycle,
        "cycle_label": cycle_label,
        "seed": seed,
        "focus_objective": focus_objective,
        "tier_c_override": tier_c_override,
        "underdetermined_streak": underdetermined_streak,
        "selected_execution_keys": selected,
        "recommended_execution_keys": recommended,
        "overall_status": overall,
        "mode": mode,
        "selection_status": selection,
        "run_dir": str(run_dir),
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(history_row) + "\n")


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
    focus_objective: str,
    resume_latest: bool = False,
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
        "--focus-objective",
        focus_objective,
        "--tier-c-justification",
        tier_c_justification.strip(),
        "--underdetermined-cycles",
        str(underdetermined_cycles),
        "--output-json",
        str(output_json),
    ]
    if run_dir is not None:
        cmd.extend(["--run-dir", str(run_dir)])
    if resume_latest:
        cmd.append("--resume-latest")
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
    parser.add_argument(
        "--focus-objective",
        type=str,
        default="ALL",
        choices=["ALL", "A", "B", "C"],
        help="Critical-path objective focus for planning/execution.",
    )
    parser.add_argument(
        "--start-fresh",
        action="store_true",
        help="Start a new RUN_* instead of resuming the latest run.",
    )
    parser.add_argument(
        "--claim-map",
        type=Path,
        default=Path("docs/physics/framework_pdf_claim_map_v1.json"),
        help="Claim map used as completion gate for framework resolution.",
    )
    parser.add_argument(
        "--require-claim-map-gate",
        action="store_true",
        help="Do not return COMPLETE unless unresolved claim count is at or below threshold.",
    )
    parser.add_argument(
        "--claim-map-target-unresolved",
        type=int,
        default=0,
        help="Required unresolved-claim threshold for completion when claim gate is enabled.",
    )
    parser.add_argument(
        "--claim-map-stall-cycles",
        type=int,
        default=2,
        help="Fail as unresolved/stalled if claim map unresolved set does not change for this many cycles.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifacts_root = args.artifacts_root.resolve()

    # Bootstrap run if needed.
    run_dir = None if args.start_fresh else latest_run_dir(artifacts_root)
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
            focus_objective=args.focus_objective,
            resume_latest=not args.start_fresh,
        )
        run_dir_value = ((bootstrap.get("result") or {}).get("run_dir") or "").strip()
        run_dir = Path(run_dir_value) if run_dir_value else latest_run_dir(artifacts_root)
        if run_dir is None:
            print("[workflow-auto-agentic] ERROR: could not initialize RUN_* directory")
            return 1

    agentic_dir = run_dir / "results" / "agentic"
    events_path = run_dir / "logs" / "agentic_events.jsonl"
    live_brief_path = run_dir / "results" / "agentic" / "live_brief.md"
    live_brief_history = run_dir / "results" / "agentic" / "live_brief_history.jsonl"

    underdetermined_streak = 0
    unresolved_streak = 0
    last_unresolved_digest = ""
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
            focus_objective=args.focus_objective,
        )
        append_event(events_path, {"ts_utc": utc_now(), "event": "planner_finished", **planner_event})

        if planner_event["worker_exit_code"] != 0:
            write_live_brief(
                path=live_brief_path,
                history_path=live_brief_history,
                run_dir=run_dir,
                cycle=cycle,
                cycle_label=cycle_label,
                seed=seed,
                focus_objective=args.focus_objective,
                tier_c_override=tier_c_enabled,
                underdetermined_streak=underdetermined_streak,
                planner_event=planner_event,
                research_event=None,
                executor_event=None,
                claim_gate=None,
            )
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
                focus_objective=args.focus_objective,
            )
            append_event(events_path, {"ts_utc": utc_now(), "event": "researcher_finished", **research_event})
            if research_event["worker_exit_code"] != 0:
                write_live_brief(
                    path=live_brief_path,
                    history_path=live_brief_history,
                    run_dir=run_dir,
                    cycle=cycle,
                    cycle_label=cycle_label,
                    seed=seed,
                    focus_objective=args.focus_objective,
                    tier_c_override=tier_c_enabled,
                    underdetermined_streak=underdetermined_streak,
                    planner_event=planner_event,
                    research_event=research_event,
                    executor_event=None,
                    claim_gate=None,
                )
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
            focus_objective=args.focus_objective,
            resume_latest=False,
        )
        append_event(events_path, {"ts_utc": utc_now(), "event": "executor_finished", **executor_event})

        if executor_event["worker_exit_code"] != 0:
            write_live_brief(
                path=live_brief_path,
                history_path=live_brief_history,
                run_dir=run_dir,
                cycle=cycle,
                cycle_label=cycle_label,
                seed=seed,
                focus_objective=args.focus_objective,
                tier_c_override=dynamic_tier_c or tier_c_enabled,
                underdetermined_streak=underdetermined_streak,
                planner_event=planner_event,
                research_event=research_event,
                executor_event=executor_event,
                claim_gate=None,
            )
            print("[workflow-auto-agentic] executor failed")
            return 1

        result = executor_event.get("result") or {}
        run_dir_value = (result.get("run_dir") or "").strip()
        if run_dir_value:
            run_dir = Path(run_dir_value)
        mode = str(result.get("mode", "UNKNOWN")).upper()
        overall = str(result.get("overall_status", "UNKNOWN")).upper()
        selection = str(result.get("selection_status", "UNKNOWN")).upper()
        effective_claim_map = refresh_claim_map_live(args.claim_map.resolve(), run_dir)
        claim_gate = claim_gate_status(effective_claim_map)
        unresolved_actions = runnable_claim_actions(claim_gate.get("unresolved_claim_ids", []))

        print(
            "[workflow-auto-agentic] "
            f"run={run_dir.name} overall={overall} mode={mode} selection={selection}"
        )

        write_live_brief(
            path=live_brief_path,
            history_path=live_brief_history,
            run_dir=run_dir,
            cycle=cycle,
            cycle_label=cycle_label,
            seed=seed,
            focus_objective=args.focus_objective,
            tier_c_override=dynamic_tier_c or tier_c_enabled,
            underdetermined_streak=underdetermined_streak,
            planner_event=planner_event,
            research_event=research_event,
            executor_event=executor_event,
            claim_gate=claim_gate,
        )

        if mode == "STOPPED" or overall == "STOPPED":
            return 1
        if overall == "COMPLETE":
            if args.require_claim_map_gate and claim_gate.get("enabled"):
                unresolved_count = int(claim_gate.get("unresolved_count", 0))
                unresolved_digest = str(claim_gate.get("unresolved_digest", ""))
                target = int(args.claim_map_target_unresolved)
                if unresolved_count <= target:
                    return 0

                if not unresolved_actions:
                    append_event(
                        events_path,
                        {
                            "ts_utc": utc_now(),
                            "event": "claim_gate_unrunnable",
                            "cycle": cycle,
                            "unresolved_count": unresolved_count,
                            "target_unresolved": target,
                            "reason": "No runnable actions for remaining unresolved claims under current tier policy.",
                        },
                    )
                    print(
                        "[workflow-auto-agentic] unresolved claims remain, but no runnable actions are available "
                        "under current tier policy; stopping unresolved"
                    )
                    return 2

                if unresolved_digest and unresolved_digest == last_unresolved_digest:
                    unresolved_streak += 1
                else:
                    unresolved_streak = 1
                    last_unresolved_digest = unresolved_digest

                append_event(
                    events_path,
                    {
                        "ts_utc": utc_now(),
                        "event": "claim_gate_unresolved",
                        "cycle": cycle,
                        "unresolved_count": unresolved_count,
                        "target_unresolved": target,
                        "unresolved_streak": unresolved_streak,
                        "runnable_actions": unresolved_actions,
                    },
                )

                if unresolved_streak >= max(1, int(args.claim_map_stall_cycles)):
                    print(
                        "[workflow-auto-agentic] claim gate stalled: unresolved claims not improving; "
                        "marking unresolved instead of COMPLETE"
                    )
                    return 2

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

                print("[workflow-auto-agentic] claim gate unresolved at max cycles")
                return 2

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
