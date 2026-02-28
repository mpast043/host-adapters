#!/usr/bin/env python3
"""Plan Framework-with-selection tests with strict Step 0 gating."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

VALID_TIERS = {"A", "B", "C"}
VALID_OBJECTIVES = {"A", "B", "C"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def choose_artifacts_root(repo_root: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    env_root = os.environ.get("HOST_ADAPTERS_EXPERIMENTAL_DATA_DIR", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    default_external = Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters")
    if default_external.exists():
        return default_external.resolve()
    return repo_root


def load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_step0_fields(test: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not _is_nonempty_str(test.get("decision_if_pass")):
        issues.append("Missing decision_if_pass")
    if not _is_nonempty_str(test.get("decision_if_fail")):
        issues.append("Missing decision_if_fail")

    objective = str(test.get("critical_path_objective", "")).strip().upper()
    if objective not in VALID_OBJECTIVES:
        issues.append("critical_path_objective must be one of A/B/C")

    tier = str(test.get("tier", "")).strip().upper()
    if tier not in VALID_TIERS:
        issues.append("tier must be one of A/B/C")
    return issues


def gate_test(
    test: dict[str, Any],
    *,
    compute_available: bool,
    allow_tier_c: bool,
    tier_c_justification: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    step0_issues = validate_step0_fields(test)
    if step0_issues:
        reasons.extend([f"STEP0_FAILURE: {x}" for x in step0_issues])
        return False, reasons

    tier = str(test["tier"]).upper()
    if tier == "A":
        pass
    elif tier == "B":
        protects = bool(test.get("protects_supported_claim", False))
        prevents = bool(test.get("prevents_regression", False))
        if not (protects or prevents):
            reasons.append("TIER_B_BLOCKED: must protect supported claim or prevent regression")
            return False, reasons
    elif tier == "C":
        if not allow_tier_c:
            reasons.append("TIER_C_BLOCKED: requires RUN_OVERRIDE_TIER_C=1")
            return False, reasons
        if not tier_c_justification.strip():
            reasons.append("TIER_C_BLOCKED: override requires one-line justification")
            return False, reasons
    else:
        reasons.append("INVALID_TIER")
        return False, reasons

    if bool(test.get("requires_compute", False)) and not compute_available:
        reasons.append("MCP_COMPUTE_REQUIRED_BUT_UNAVAILABLE")
        return False, reasons

    return True, reasons


def rank_key(test: dict[str, Any]) -> tuple[int, int, str]:
    tier_priority = {"A": 0, "B": 1, "C": 2}
    objective_priority = {"A": 0, "B": 1, "C": 2}
    tier = str(test["tier"]).upper()
    objective = str(test["critical_path_objective"]).upper()
    minutes = int(test.get("estimated_minutes", 999))
    return (tier_priority[tier], objective_priority[objective], minutes)


def plan_tests(
    *,
    tests: list[dict[str, Any]],
    max_minutes: int,
    max_runs: int,
    compute_available: bool,
    allow_tier_c: bool,
    tier_c_justification: str,
) -> dict[str, Any]:
    proposed = [dict(t) for t in tests]
    for t in proposed:
        t["tier"] = str(t.get("tier", "")).upper()
        t["critical_path_objective"] = str(t.get("critical_path_objective", "")).upper()

    allowed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for t in proposed:
        ok, reasons = gate_test(
            t,
            compute_available=compute_available,
            allow_tier_c=allow_tier_c,
            tier_c_justification=tier_c_justification,
        )
        t["gate"] = {"allowed": ok, "reasons": reasons}
        if ok:
            allowed.append(t)
        else:
            blocked.append(t)

    allowed_sorted = sorted(allowed, key=rank_key)

    selected: list[dict[str, Any]] = []
    remaining_minutes = max_minutes
    for t in allowed_sorted:
        if len(selected) >= max_runs:
            break
        est = int(t.get("estimated_minutes", 0))
        if est <= remaining_minutes:
            selected.append(t)
            remaining_minutes -= est

    unselected_allowed = [t for t in allowed_sorted if t not in selected]
    has_non_c_queued = any(t["tier"] in {"A", "B"} for t in unselected_allowed)
    has_c_queued = any(t["tier"] == "C" for t in unselected_allowed) or any(t["tier"] == "C" for t in blocked)

    stop_condition_triggered = False
    stop_condition_note = ""
    if remaining_minutes > 0 and has_c_queued and not has_non_c_queued:
        stop_condition_triggered = True
        stop_condition_note = (
            "Time remained while only Tier C work was queued. "
            "Budget reallocated to Tier A hardening tasks."
        )
        hardening = [
            t
            for t in allowed_sorted
            if t["tier"] == "A" and bool(t.get("hardening_task", False)) and t not in selected
        ]
        for t in hardening:
            if len(selected) >= max_runs:
                break
            est = int(t.get("estimated_minutes", 0))
            if est <= remaining_minutes:
                selected.append(t)
                remaining_minutes -= est

    return {
        "selected": selected,
        "proposals": proposed,
        "blocked": blocked,
        "remaining_minutes": remaining_minutes,
        "stop_condition": {
            "triggered": stop_condition_triggered,
            "note": stop_condition_note,
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    selected = plan["selected"]
    blocked = plan["blocked"]
    lines = [
        "# Framework Selection Test Plan",
        "",
        f"Generated: {plan['generated_at_utc']}",
        f"Mode: `{plan['mode']}`",
        f"Compute target: `{plan['compute_target'] or 'none'}`",
        "",
        "## Step 0 Gate",
        "1. What decision changes if this test passes vs fails?",
        "2. Which critical path objective does it serve?",
        "",
        "Tier rules applied:",
        "- Tier A: always allowed; highest priority.",
        "- Tier B: allowed only when it protects a supported claim or prevents regression.",
        "- Tier C: blocked unless `RUN_OVERRIDE_TIER_C=1` with one-line justification.",
        "",
        "## Selected",
        "| Test | Tier | Objective | Minutes |",
        "|---|---|---|---:|",
    ]
    if selected:
        for t in selected:
            lines.append(
                f"| {t['test_id']} | {t['tier']} | {t['critical_path_objective']} | {int(t.get('estimated_minutes', 0))} |"
            )
    else:
        lines.append("| none | - | - | 0 |")

    lines.extend(["", "## Blocked", "| Test | Tier | Reason |", "|---|---|---|"])
    if blocked:
        for t in blocked:
            reason = "; ".join(t["gate"]["reasons"]) or "blocked"
            lines.append(f"| {t['test_id']} | {t.get('tier', '?')} | {reason} |")
    else:
        lines.append("| none | - | - |")

    stop_condition = plan["stop_condition"]
    lines.extend(
        [
            "",
            "## Stop Condition",
            f"- Triggered: `{stop_condition['triggered']}`",
            f"- Note: {stop_condition['note'] or 'Not triggered'}",
            "",
            f"- Remaining budget minutes: `{plan['remaining_minutes']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan framework-selection tests with Step 0 gate enforcement")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifacts-root", type=Path, default=None)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/physics/framework_selection_test_catalog_v1.json"),
    )
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--max-runs", type=int, default=80)
    parser.add_argument("--compute-target", type=str, default="")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--tier-c-justification", type=str, default="")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/state/framework_selection_plan_latest.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/state/framework_selection_plan_latest.md"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifacts_root = choose_artifacts_root(repo_root, args.artifacts_root)

    catalog_path = args.catalog if args.catalog.is_absolute() else repo_root / args.catalog
    catalog = load_catalog(catalog_path)

    allow_tier_c = os.environ.get("RUN_OVERRIDE_TIER_C", "").strip() == "1"
    compute_available = bool(args.compute_target.strip()) and not args.local_only

    plan_core = plan_tests(
        tests=catalog.get("tests", []),
        max_minutes=args.max_minutes,
        max_runs=args.max_runs,
        compute_available=compute_available,
        allow_tier_c=allow_tier_c,
        tier_c_justification=args.tier_c_justification,
    )

    plan = {
        "generated_at_utc": utc_now(),
        "catalog_path": str(catalog_path),
        "mode": "LOCAL_ONLY" if args.local_only else "FULL",
        "compute_target": args.compute_target.strip() or None,
        "tier_c_override": {
            "enabled": allow_tier_c,
            "justification": args.tier_c_justification.strip() or None,
        },
        "proposals": plan_core["proposals"],
        "selected": plan_core["selected"],
        "blocked": plan_core["blocked"],
        "remaining_minutes": plan_core["remaining_minutes"],
        "stop_condition": plan_core["stop_condition"],
    }

    out_json = args.output_json if args.output_json.is_absolute() else artifacts_root / args.output_json
    out_md = args.output_md if args.output_md.is_absolute() else artifacts_root / args.output_md
    write_json(out_json, plan)
    write_markdown(out_md, plan)
    print(str(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
