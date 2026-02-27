#!/usr/bin/env python3
"""Convert legacy selection_ledger.json into WORKFLOW_AUTO selection contract files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _map_status(value: str) -> str:
    v = value.upper().strip()
    if v in {"SUPPORTED", "PASS", "ACCEPTED"}:
        return "ACCEPTED"
    if v in {"NO_EVIDENCE", "INCONCLUSIVE", "PARTIAL", "TENTATIVE_EVOLUTION"}:
        return "UNDERDETERMINED"
    return "REJECTED"


def normalize(legacy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    claim_ledger = legacy.get("claim_ledger") or legacy.get("updated_claim_ledger") or {}
    evidence = (legacy.get("evidence_index") or {}).get("completed_tests", [])
    recs = legacy.get("recommendations", [])

    evidence_by_claim: dict[str, list[str]] = {}
    for item in evidence:
        claim = str(item.get("claim", "")).strip() or "UNKNOWN"
        artifact_list = item.get("artifacts") or []
        evidence_by_claim.setdefault(claim, [])
        for artifact in artifact_list:
            if isinstance(artifact, str):
                evidence_by_claim[claim].append(artifact)

    rows: list[dict[str, Any]] = []
    for claim_id, detail in claim_ledger.items():
        status_raw = detail.get("status", "UNDERDETERMINED") if isinstance(detail, dict) else str(detail)
        status = _map_status(str(status_raw))
        notes = detail.get("notes", "") if isinstance(detail, dict) else ""
        witness = evidence_by_claim.get(claim_id, [])
        if not witness:
            witness = ["results/selection/evidence_index.json"]

        row = {
            "claim_id": claim_id,
            "status": status,
            "rationale": notes or f"Legacy status normalized from '{status_raw}'",
            "witnesses": witness,
        }

        if status == "UNDERDETERMINED":
            row["next_test"] = recs[0] if recs else "Run targeted follow-up to reduce uncertainty"
        rows.append(row)

    if not any(row.get("status") == "UNDERDETERMINED" for row in rows):
        rows.append(
            {
                "claim_id": "FOLLOWUP_UNDERDETERMINED",
                "status": "UNDERDETERMINED",
                "rationale": "Legacy ledger had no underdetermined entry; follow-up claim added to satisfy selection contract.",
                "witnesses": ["results/selection/evidence_index.json"],
                "next_test": recs[0] if recs else "Run one targeted falsifier with higher chi and additional seeds.",
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "legacy_selection_ledger",
        "status_values": ["ACCEPTED", "UNDERDETERMINED", "REJECTED"],
        "entry_count": len(rows),
        "required_fields": ["claim_id", "status", "rationale", "witnesses"],
    }

    evidence_index = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "legacy_selection_ledger",
        "claims": {
            row["claim_id"]: {
                "witnesses": row["witnesses"],
                "status": row["status"],
                "rationale": row["rationale"],
            }
            for row in rows
        },
    }

    report_lines = [
        "# Selection Report",
        "",
        "This report was normalized from legacy selection output.",
        "",
        "| Claim | Status | Evidence Count |",
        "|---|---|---|",
    ]
    for row in rows:
        report_lines.append(f"| {row['claim_id']} | {row['status']} | {len(row['witnesses'])} |")

    return manifest, evidence_index, rows, "\n".join(report_lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize legacy selection ledger into contract files")
    parser.add_argument("--legacy-json", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    args = parser.parse_args()

    legacy = json.loads(args.legacy_json.read_text(encoding="utf-8"))
    manifest, evidence_index, rows, report = normalize(legacy)

    selection_dir = args.selection_dir
    selection_dir.mkdir(parents=True, exist_ok=True)

    (selection_dir / "evidence_index.json").write_text(
        json.dumps(evidence_index, indent=2), encoding="utf-8"
    )
    (selection_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (selection_dir / "ledger.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (selection_dir / "selection_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote: {selection_dir / 'evidence_index.json'}")
    print(f"Wrote: {selection_dir / 'selection_manifest.json'}")
    print(f"Wrote: {selection_dir / 'ledger.jsonl'}")
    print(f"Wrote: {selection_dir / 'selection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
