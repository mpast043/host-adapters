#!/usr/bin/env python3
"""Validate a RUN_* directory against WORKFLOW_AUTO artifact contract.

This checker enforces required paths, selection ledger completeness,
and normalized workflow status schema compatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_LEDGER_STATUS = {"ACCEPTED", "UNDERDETERMINED", "REJECTED"}


@dataclass
class Finding:
    code: str
    level: str
    message: str


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _exists(path: Path, findings: list[Finding], code: str) -> bool:
    if not path.exists():
        findings.append(Finding(code=code, level="ERROR", message=f"Missing: {path}"))
        return False
    return True


def _validate_status_keys(status_obj: dict[str, Any], findings: list[Finding]) -> None:
    required_keys = {
        "run_id",
        "mode",
        "mcp_targets",
        "contract_status",
        "science_status",
        "selection_status",
        "retention_status",
        "artifact_index",
    }
    missing = sorted(required_keys - set(status_obj.keys()))
    if missing:
        findings.append(
            Finding(
                code="STATUS_KEYS_MISSING",
                level="ERROR",
                message=f"results/workflow_auto_status.json missing keys: {missing}",
            )
        )


def _validate_index_paths(run_dir: Path, findings: list[Finding]) -> None:
    index_path = run_dir / "results/index.json"
    if not _exists(index_path, findings, "INDEX_MISSING"):
        return

    index_obj = _load_json(index_path)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and isinstance(value["path"], str):
                candidate = (run_dir / value["path"]).resolve()
                if not candidate.exists():
                    findings.append(
                        Finding(
                            code="INDEX_PATH_INVALID",
                            level="ERROR",
                            message=f"Index path missing: {value['path']}",
                        )
                    )
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(index_obj)


def _validate_selection_ledger(run_dir: Path, findings: list[Finding]) -> None:
    selection_dir = run_dir / "results/selection"
    if not _exists(selection_dir / "selection_manifest.json", findings, "SELECTION_MANIFEST_MISSING"):
        return
    if not _exists(selection_dir / "selection_report.md", findings, "SELECTION_REPORT_MISSING"):
        return
    ledger_path = selection_dir / "ledger.jsonl"
    if not _exists(ledger_path, findings, "SELECTION_LEDGER_MISSING"):
        return

    lines = [line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        findings.append(Finding(code="SELECTION_LEDGER_EMPTY", level="ERROR", message="Selection ledger has no entries"))
        return

    has_underdetermined_next_test = False

    for i, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append(
                Finding(
                    code="SELECTION_LEDGER_JSON_INVALID",
                    level="ERROR",
                    message=f"Invalid JSON at ledger line {i}: {exc}",
                )
            )
            continue

        for key in ("claim_id", "status", "rationale", "witnesses"):
            if key not in row:
                findings.append(
                    Finding(
                        code="SELECTION_LEDGER_FIELD_MISSING",
                        level="ERROR",
                        message=f"Line {i} missing required field: {key}",
                    )
                )

        status = row.get("status")
        if status not in ALLOWED_LEDGER_STATUS:
            findings.append(
                Finding(
                    code="SELECTION_LEDGER_STATUS_INVALID",
                    level="ERROR",
                    message=f"Line {i} has invalid status '{status}'",
                )
            )

        witnesses = row.get("witnesses")
        if not isinstance(witnesses, list) or not witnesses:
            findings.append(
                Finding(
                    code="SELECTION_LEDGER_WITNESS_EMPTY",
                    level="ERROR",
                    message=f"Line {i} must have at least one witness pointer",
                )
            )

        if status == "UNDERDETERMINED" and isinstance(row.get("next_test"), str) and row["next_test"].strip():
            has_underdetermined_next_test = True

    if not has_underdetermined_next_test:
        findings.append(
            Finding(
                code="SELECTION_UNDERDETERMINED_MISSING_NEXT_TEST",
                level="ERROR",
                message="No UNDERDETERMINED ledger item includes next_test recommendation",
            )
        )


def validate_run(run_dir: Path, schema_path: Path) -> tuple[bool, list[Finding], dict[str, Any]]:
    findings: list[Finding] = []

    required_root = [
        run_dir / "logs",
        run_dir / "results",
        run_dir / "tmp",
        run_dir / "manifest.json",
        run_dir / "summary.md",
        run_dir / "RETAINED.txt",
    ]
    for req in required_root:
        _exists(req, findings, "ROOT_REQUIRED_MISSING")

    required_results = [
        run_dir / "results/VERDICT.json",
        run_dir / "results/index.json",
        run_dir / "results/workflow_auto_status.json",
        run_dir / "results/science/run_plan.json",
        run_dir / "results/science/baseline_summary.json",
        run_dir / "results/science/exploration_proposals.json",
        run_dir / "results/science/exploration_selected.json",
        run_dir / "results/science/campaign/campaign_index.csv",
        run_dir / "results/science/campaign/campaign_report.md",
        run_dir / "results/science/campaign/model_comparison.json",
        run_dir / "results/selection/evidence_index.json",
        run_dir / "results/selection/selection_manifest.json",
        run_dir / "results/selection/ledger.jsonl",
        run_dir / "results/selection/selection_report.md",
    ]
    for req in required_results:
        _exists(req, findings, "RESULT_REQUIRED_MISSING")

    status_path = run_dir / "results/workflow_auto_status.json"
    if status_path.exists():
        status_obj = _load_json(status_path)
        _validate_status_keys(status_obj, findings)

        # Optional jsonschema validation if available.
        if schema_path.exists():
            try:
                import jsonschema  # type: ignore

                schema_obj = _load_json(schema_path)
                jsonschema.validate(instance=status_obj, schema=schema_obj)
            except ImportError:
                findings.append(
                    Finding(
                        code="JSONSCHEMA_NOT_INSTALLED",
                        level="WARN",
                        message="jsonschema not installed; key-level validation applied only",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                findings.append(
                    Finding(
                        code="STATUS_SCHEMA_VALIDATION_FAILED",
                        level="ERROR",
                        message=f"Schema validation failed: {exc}",
                    )
                )

    _validate_index_paths(run_dir, findings)
    _validate_selection_ledger(run_dir, findings)

    errors = [f for f in findings if f.level == "ERROR"]
    summary = {
        "run_dir": str(run_dir),
        "schema": str(schema_path),
        "errors": len(errors),
        "warnings": len([f for f in findings if f.level == "WARN"]),
        "ready": len(errors) == 0,
        "findings": [f.__dict__ for f in findings],
    }
    return len(errors) == 0, findings, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a RUN_* directory against WORKFLOW_AUTO contract")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to RUN_YYYYMMDD_HHMMSS directory")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("tools/workflow_auto_schema_v1.json"),
        help="Path to workflow status JSON schema",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    schema_path = args.schema.resolve() if not args.schema.is_absolute() else args.schema

    ok, findings, summary = validate_run(run_dir, schema_path)

    print(f"Validation target: {run_dir}")
    print(f"Schema: {schema_path}")
    for finding in findings:
        print(f"[{finding.level}] {finding.code}: {finding.message}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote report: {args.output}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
