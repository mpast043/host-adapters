#!/usr/bin/env python3
"""OpenClaw optimization readiness check.

Validates local OpenClaw memory + compaction + skill readiness and writes a
machine-readable report for automation gates.
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


def run_json(cmd: list[str]) -> Any:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    text = proc.stdout.strip()
    if not text:
        return None
    return json.loads(text)


def choose_output_path(explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output.resolve()
    env_root = os.environ.get("HOST_ADAPTERS_EXPERIMENTAL_DATA_DIR", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        return root / "openclaw_adapter_data" / "openclaw_opt_check_latest.json"
    default_root = Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters")
    if default_root.exists():
        return default_root / "openclaw_adapter_data" / "openclaw_opt_check_latest.json"
    return Path("openclaw_opt_check_latest.json").resolve()


def add_finding(findings: list[dict[str, str]], level: str, code: str, message: str) -> None:
    findings.append({"level": level, "code": code, "message": message})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OpenClaw optimization readiness")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--min-files", type=int, default=10)
    parser.add_argument("--min-chunks", type=int, default=100)
    parser.add_argument("--min-eligible-skills", type=int, default=12)
    parser.add_argument("--max-disabled-skills", type=int, default=2)
    parser.add_argument("--require-compaction-mode", type=str, default="safeguard")
    parser.add_argument("--require-memory-enabled", action="store_true", default=True)
    parser.add_argument("--require-hybrid-search", action="store_true", default=True)
    args = parser.parse_args()

    output_path = choose_output_path(args.output)
    findings: list[dict[str, str]] = []

    memory_status_raw = run_json(["openclaw", "memory", "status", "--json"])
    if not isinstance(memory_status_raw, list) or not memory_status_raw:
        raise RuntimeError("Unexpected output from openclaw memory status --json")
    memory_status = memory_status_raw[0].get("status", {})

    skills_check = run_json(["openclaw", "skills", "check", "--json"])
    if not isinstance(skills_check, dict):
        raise RuntimeError("Unexpected output from openclaw skills check --json")
    summary = skills_check.get("summary", {})

    compaction_mode = run_json(["openclaw", "config", "get", "--json", "agents.defaults.compaction.mode"])
    memory_cfg = run_json(["openclaw", "config", "get", "--json", "agents.defaults.memorySearch"])

    files = int(memory_status.get("files", 0) or 0)
    chunks = int(memory_status.get("chunks", 0) or 0)
    provider = str(memory_status.get("provider", "none"))
    search_mode = str(memory_status.get("custom", {}).get("searchMode", "unknown"))
    eligible = int(summary.get("eligible", 0) or 0)
    disabled = int(summary.get("disabled", 0) or 0)

    if args.require_memory_enabled and not bool(memory_cfg.get("enabled", False)):
        add_finding(findings, "ERROR", "MEMORY_DISABLED", "agents.defaults.memorySearch.enabled is false")
    if compaction_mode != args.require_compaction_mode:
        add_finding(
            findings,
            "ERROR",
            "COMPACTION_MODE_MISMATCH",
            f"Compaction mode is '{compaction_mode}', expected '{args.require_compaction_mode}'",
        )
    if provider == "none":
        add_finding(findings, "ERROR", "MEMORY_PROVIDER_NONE", "Memory provider is none; semantic recall is unavailable")
    if files < args.min_files:
        add_finding(findings, "ERROR", "MEMORY_FILES_LOW", f"Indexed files {files} < required {args.min_files}")
    if chunks < args.min_chunks:
        add_finding(findings, "ERROR", "MEMORY_CHUNKS_LOW", f"Indexed chunks {chunks} < required {args.min_chunks}")
    if args.require_hybrid_search and search_mode != "hybrid":
        add_finding(
            findings,
            "ERROR",
            "MEMORY_SEARCH_MODE_NOT_HYBRID",
            f"searchMode is '{search_mode}', expected 'hybrid'",
        )
    if eligible < args.min_eligible_skills:
        add_finding(
            findings,
            "ERROR",
            "SKILLS_ELIGIBLE_LOW",
            f"Eligible skills {eligible} < required {args.min_eligible_skills}",
        )
    if disabled > args.max_disabled_skills:
        add_finding(
            findings,
            "WARN",
            "SKILLS_DISABLED_HIGH",
            f"Disabled skills {disabled} > threshold {args.max_disabled_skills}",
        )

    errors = sum(1 for f in findings if f["level"] == "ERROR")
    warnings = sum(1 for f in findings if f["level"] == "WARN")
    score = max(0, 100 - (errors * 20) - (warnings * 5))
    ready = errors == 0

    report = {
        "generated_at_utc": utc_now(),
        "ready": ready,
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "memory": {
                "provider": provider,
                "files": files,
                "chunks": chunks,
                "searchMode": search_mode,
                "sources": memory_status.get("sources", []),
            },
            "compaction": {"mode": compaction_mode},
            "skills": {
                "eligible": eligible,
                "disabled": disabled,
                "missingRequirements": int(summary.get("missingRequirements", 0) or 0),
            },
        },
        "thresholds": {
            "min_files": args.min_files,
            "min_chunks": args.min_chunks,
            "min_eligible_skills": args.min_eligible_skills,
            "max_disabled_skills": args.max_disabled_skills,
            "require_compaction_mode": args.require_compaction_mode,
            "require_hybrid_search": args.require_hybrid_search,
            "require_memory_enabled": args.require_memory_enabled,
        },
        "findings": findings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote report: {output_path}")
    print(
        "OPENCLAW_OPT_CHECK ready={ready} score={score}/100 errors={errors} warnings={warnings}".format(
            ready=ready,
            score=score,
            errors=errors,
            warnings=warnings,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
