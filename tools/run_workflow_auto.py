#!/usr/bin/env python3
"""Deterministic WORKFLOW_AUTO orchestrator.

Implements Steps 0-6 from WORKFLOW_AUTO.md with strict artifact paths,
LOCAL_ONLY downscoping when compute MCP is unavailable, machine-readable
event emission, and retention packaging.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PORT_ROTATION = [8080, 18080, 28080, 38080]
FAILURE_CODES = {
    "LINT_FAILURE",
    "TEST_FAILURE",
    "CONTRACT_FAILURE",
    "RUNTIME_FAILURE",
    "SCIENCE_EVIDENCE_FAILURE",
    "SELECTION_FAILURE",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_result_with_history(
    *,
    results_dir: Path,
    filename: str,
    obj: Any,
    freeze_legacy: bool = True,
) -> dict[str, str]:
    """Write append-only snapshot + latest pointer, with optional legacy freeze.

    - Always writes `results/history/<stem>_<timestamp>.json`
    - Always writes `results/<stem>_latest.json`
    - Writes/updates legacy `results/<filename>` only if it does not exist, or
      when `freeze_legacy` is False.
    """
    target = results_dir / filename
    stem = target.stem
    suffix = target.suffix
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

    history_dir = results_dir / "history"
    snapshot = history_dir / f"{stem}_{ts}{suffix}"
    latest = results_dir / f"{stem}_latest{suffix}"

    legacy_preexists = target.exists()
    legacy_updated = (not legacy_preexists) or (not freeze_legacy)

    write_json(snapshot, obj)
    write_json(latest, obj)
    if legacy_updated:
        write_json(target, obj)

    index_row = {
        "ts_utc": utc_now(),
        "artifact": filename,
        "snapshot_path": str(snapshot.relative_to(results_dir.parent)),
        "latest_path": str(latest.relative_to(results_dir.parent)),
        "legacy_path": str(target.relative_to(results_dir.parent)),
        "legacy_updated": legacy_updated,
    }
    index_path = history_dir / "history_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_row) + "\n")

    return {
        "snapshot_path": str(snapshot.relative_to(results_dir.parent)),
        "latest_path": str(latest.relative_to(results_dir.parent)),
        "legacy_path": str(target.relative_to(results_dir.parent)),
    }


def load_result_json(results_dir: Path, filename: str) -> dict[str, Any] | None:
    target = results_dir / filename
    latest = results_dir / f"{target.stem}_latest{target.suffix}"
    for candidate in (latest, target):
        if candidate.exists():
            return safe_load_json(candidate)
    return None


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_latest_subdir(path: Path) -> Path | None:
    if not path.exists() or not path.is_dir():
        return None
    children = [p for p in path.iterdir() if p.is_dir()]
    if not children:
        return None
    return sorted(children, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def tail_lines(path: Path, n: int = 200) -> str:
    lines = read_text(path).splitlines()
    return "\n".join(lines[-n:]) + ("\n" if lines else "")


def http_health(endpoint: str, timeout_s: float = 2.0) -> tuple[bool, int | None, str]:
    url = f"{endpoint.rstrip('/')}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
            return True, resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # A reachable server that lacks /health may return 404; treat as alive.
        if exc.code == 404:
            return True, exc.code, body
        return False, exc.code, body
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def get_python(repo_root: Path) -> str:
    venv_py = repo_root / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else "python3"


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _event_writer(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def record_event(events_path: Path, step: str, event_type: str, payload: dict[str, Any]) -> None:
    _event_writer(
        events_path,
        {
            "ts_utc": utc_now(),
            "step": step,
            "event_type": event_type,
            **payload,
        },
    )


def run_command(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    events_path: Path,
    step: str,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.txt"

    # Resume mode: reuse successful prior command results by step/name to avoid re-executing work.
    if bool(manifest.get("_resume_enabled")):
        for prior in reversed(manifest.get("commands", [])):
            if prior.get("step") != step or prior.get("name") != name:
                continue
            if list(prior.get("command", [])) != command:
                continue
            if int(prior.get("exit_code", 1)) != 0:
                continue
            prior_log = run_dir / str(prior.get("log_path", ""))
            if not prior_log.exists():
                continue
            record_event(
                events_path,
                step,
                "command_reused",
                {
                    "name": name,
                    "command": " ".join(command),
                    "log_path": str(prior.get("log_path")),
                },
            )
            return prior

    started = time.time()
    started_iso = utc_now()
    rc = 1

    record_event(
        events_path,
        step,
        "command_started",
        {
            "name": name,
            "command": " ".join(command),
            "cwd": str(cwd),
            "log_path": str(log_path.relative_to(run_dir)),
        },
    )

    with log_path.open("w", encoding="utf-8") as fout:
        try:
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=fout,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            rc = proc.returncode
        except FileNotFoundError as exc:
            fout.write(f"FileNotFoundError: {exc}\n")
            rc = 127
        except Exception as exc:  # noqa: BLE001
            fout.write(f"Unhandled exception while executing command: {exc}\n")
            rc = 1

    ended = time.time()
    ended_iso = utc_now()

    entry = {
        "name": name,
        "step": step,
        "command": command,
        "cwd": str(cwd),
        "exit_code": rc,
        "started_utc": started_iso,
        "ended_utc": ended_iso,
        "duration_s": round(ended - started, 3),
        "log_path": str(log_path.relative_to(run_dir)),
    }
    manifest.setdefault("commands", []).append(entry)

    if rc != 0:
        tail_path = logs_dir / f"{name}_tail200.txt"
        tail_path.write_text(tail_lines(log_path, n=200), encoding="utf-8")
        entry["tail200_path"] = str(tail_path.relative_to(run_dir))

    record_event(
        events_path,
        step,
        "command_finished",
        {
            "name": name,
            "exit_code": rc,
            "duration_s": round(ended - started, 3),
            "log_path": str(log_path.relative_to(run_dir)),
        },
    )

    return entry


def parse_mcporter_targets(mcporter_list_text: str) -> tuple[str | None, str | None, list[str]]:
    compute_target = None
    docs_target = None
    discovered: list[str] = []

    for line in mcporter_list_text.splitlines():
        l = line.strip()
        if not l:
            continue
        if l.lower().startswith("name") or l.lower().startswith("---"):
            continue

        # mcporter list lines are typically bullet-form:
        # - server-name (N tools, ...)
        bullet_match = re.match(r"^-\s+([^\s(]+)", l)
        token = bullet_match.group(1) if bullet_match else None
        if token:
            discovered.append(token)

        ll = l.lower()
        has_tools_count = bool(re.search(r"\(\s*\d+\s+tools?\b", ll))
        has_bad_status = any(word in ll for word in ["offline", "auth required", "error", "failed"])
        online_like = (any(word in ll for word in ["online", "connected", "healthy", "running"]) or has_tools_count) and not has_bad_status
        if not online_like:
            continue

        if compute_target is None and any(k in ll for k in ["compute", "cpu", "gpu", "executor"]):
            compute_target = token or l
        if docs_target is None and any(k in ll for k in ["docs", "document", "notion", "drive", "pdf"]):
            docs_target = token or l

    return compute_target, docs_target, sorted(set(discovered))


def discover_capabilities(repo_root: Path, discovered_servers: list[str]) -> dict[str, Any]:
    commands = [
        "python3",
        "mcporter",
        "openclaw",
        "ruff",
        "pytest",
        "pdftotext",
        "pdftoppm",
        "uv",
    ]
    command_status = {cmd: {"available": shutil.which(cmd) is not None, "path": shutil.which(cmd)} for cmd in commands}

    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    skills_root = codex_home / "skills"
    skills: list[dict[str, str]] = []
    if skills_root.exists():
        for child in sorted(skills_root.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                skills.append({"name": child.name, "path": str(skill_md)})

    return {
        "generated_at_utc": utc_now(),
        "repo_root": str(repo_root),
        "commands": command_status,
        "mcp_servers_discovered": discovered_servers,
        "skills_root": str(skills_root),
        "skills": skills,
    }


def find_pdf(repo_root: Path, explicit_pdf: Path | None) -> Path | None:
    if explicit_pdf and explicit_pdf.exists():
        return explicit_pdf.resolve()

    names = ["Framework with selection.pdf", "Framework with Selection.pdf"]
    for name in names:
        candidate = repo_root / name
        if candidate.exists():
            return candidate.resolve()

    for folder in [repo_root / "docs", repo_root / "spec"]:
        if not folder.exists():
            continue
        for name in names:
            candidate = folder / name
            if candidate.exists():
                return candidate.resolve()

    external_default = Path("/Users/meganpastore/Clawdbot/Repos/capacity-demo/Framework with selection.pdf")
    if external_default.exists():
        return external_default.resolve()

    return None


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


def safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def find_latest_run_dir(artifacts_root: Path) -> Path | None:
    if not artifacts_root.exists():
        return None
    runs = [p for p in artifacts_root.glob("RUN_*") if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def locate_baseline_runners(repo_root: Path) -> dict[str, Path]:
    candidates = {
        "claim2": [
            repo_root / "prototype/experiments/exp2_mera_tradeoff/exp2_mera_tradeoff.py",
        ],
        "claim3": [
            repo_root / "experiments/claim3/exp3_claim3_optionB_runner.py",
            repo_root / "experiments/claim3/exp3_claim3_entanglement_max_mincut_runner.py",
        ],
        "claim3p": [
            repo_root / "experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py",
        ],
    }

    resolved: dict[str, Path] = {}
    for key, options in candidates.items():
        hit = next((p for p in options if p.exists()), None)
        if hit is None:
            raise FileNotFoundError(f"Missing baseline runner for {key}; looked in: {options}")
        resolved[key] = hit
    return resolved


def detect_science_verdict(artifact_path: Path) -> str:
    verdict_file = artifact_path / "verdict.json"
    if not verdict_file.exists():
        return "NOT_RUN"
    try:
        verdict_obj = json.loads(verdict_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "NOT_RUN"

    raw = str(verdict_obj.get("verdict", "")).upper().strip()
    if raw in {"SUPPORTED", "PASS"}:
        return "SUPPORTED"
    if raw in {"REJECTED", "FAIL"}:
        return "REJECTED"
    if raw in {"INCONCLUSIVE", "PARTIAL", "NOT_SUPPORTED", "NEEDS_REVIEW"}:
        return "INCONCLUSIVE"
    return "NOT_RUN"


def build_selection_outputs(
    *,
    run_dir: Path,
    pdf_path: Path,
    evidence_index: dict[str, Any],
    contract_status: str,
    science_status: str,
    science_artifacts: dict[str, str],
    campaign_rows: list[dict[str, Any]],
    focus_objective: str,
) -> tuple[str, list[dict[str, Any]]]:
    selection_dir = run_dir / "results" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    contract_row = {
        "claim_id": "CGR_CONTRACT_COMPLIANCE",
        "status": "ACCEPTED" if contract_status in {"PASS", "LINT_DEBT"} else "REJECTED",
        "rationale": "Contract + SDK validation status mapped from workflow contract checks.",
        "witnesses": [
            "results/contracts/contract_result.json",
            "results/sdk_validation/sdk_validation_result.json",
        ],
    }
    rows.append(contract_row)

    focus = (focus_objective or "ALL").strip().upper()
    if focus == "A":
        latest_by_test: dict[str, dict[str, Any]] = {}
        for row in campaign_rows:
            test_id = str(row.get("test_id", "")).strip()
            if test_id:
                latest_by_test[test_id] = row

        required_tests = ["platform_mcporter_health_snapshot", "platform_openclaw_opt_check"]
        missing_required = [t for t in required_tests if t not in latest_by_test]

        def _is_fail(verdict: str) -> bool:
            return verdict.upper() in {"FAIL", "REJECTED"}

        def _is_pass(verdict: str) -> bool:
            return verdict.upper() in {"PASS", "SUPPORTED", "ACCEPTED"}

        failed_tests = [t for t in required_tests if t in latest_by_test and _is_fail(str(latest_by_test[t].get("verdict", "")))]
        alt_map = {"platform_openclaw_opt_check": "platform_openclaw_opt_check_relaxed"}
        pending_alternatives = [alt_map[t] for t in failed_tests if alt_map.get(t) and alt_map[t] not in latest_by_test]

        platform_status = "UNDERDETERMINED"
        rationale = ""
        next_test = ""
        if missing_required:
            rationale = f"Missing required Tier A checks: {', '.join(missing_required)}."
            next_test = f"Run missing Tier A checks: {', '.join(missing_required)}."
        elif pending_alternatives:
            rationale = (
                "Tier A strict checks failed and alternative diagnostics were not executed yet."
            )
            next_test = f"Run Tier A alternatives: {', '.join(pending_alternatives)}."
        elif failed_tests:
            platform_status = "REJECTED"
            rationale = (
                "Tier A readiness checks are conclusive and failing; platform is not ready for autonomous science expansion."
            )
            next_test = "Apply remediation from finding codes and re-run Tier A checks."
        elif all(_is_pass(str(latest_by_test[t].get("verdict", ""))) for t in required_tests):
            platform_status = "ACCEPTED"
            rationale = "Tier A readiness checks passed and are conclusive for platform-governance gating."
            next_test = ""
        else:
            rationale = "Tier A checks produced non-conclusive verdicts."
            next_test = "Re-run Tier A checks with diagnostics enabled."

        platform_witnesses = []
        for t in required_tests:
            row = latest_by_test.get(t)
            if row and row.get("artifact_path"):
                platform_witnesses.append(str(row["artifact_path"]))
        for t in ("platform_openclaw_opt_check_relaxed",):
            row = latest_by_test.get(t)
            if row and row.get("artifact_path"):
                platform_witnesses.append(str(row["artifact_path"]))
        if not platform_witnesses:
            platform_witnesses = ["results/science/campaign/campaign_index.csv"]

        platform_row: dict[str, Any] = {
            "claim_id": "PLATFORM_READINESS_TIER_A",
            "status": platform_status,
            "rationale": rationale,
            "witnesses": platform_witnesses,
        }
        if next_test:
            platform_row["next_test"] = next_test
        rows.append(platform_row)

        rows.append(
            {
                "claim_id": "PDF_FRAMEWORK_TRACEABILITY",
                "status": "ACCEPTED" if platform_status in {"ACCEPTED", "REJECTED"} else "UNDERDETERMINED",
                "rationale": "PDF source is retained and linked to local executable witnesses for Tier A readiness.",
                "witnesses": [str(pdf_path), "results/selection/evidence_index.json"],
            }
        )
    elif focus == "B":
        latest_by_test: dict[str, dict[str, Any]] = {}
        for row in campaign_rows:
            test_id = str(row.get("test_id", "")).strip()
            if test_id:
                latest_by_test[test_id] = row

        def _status_for(test_id: str) -> tuple[str, str]:
            row = latest_by_test.get(test_id)
            if row is None:
                return "UNDERDETERMINED", f"Missing test result for {test_id}."
            verdict = str(row.get("verdict", "")).upper()
            if verdict in {"PASS", "SUPPORTED", "ACCEPTED"}:
                return "ACCEPTED", f"{test_id} passed."
            if verdict in {"FAIL", "REJECTED"}:
                return "REJECTED", f"{test_id} failed."
            return "UNDERDETERMINED", f"{test_id} produced non-conclusive verdict '{verdict}'."

        claim2_status, claim2_reason = _status_for("claim2_seed_perturbation")
        claim3_status, claim3_reason = _status_for("claim3_optionb_regime_check")

        rows.append(
            {
                "claim_id": "CLAIM_2_REGRESSION_TIER_B",
                "status": claim2_status,
                "rationale": claim2_reason,
                "witnesses": ["results/science/campaign/campaign_index.csv"],
                "next_test": "Expand Claim 2 seed sweep and compare stability envelopes." if claim2_status != "ACCEPTED" else "",
            }
        )
        rows.append(
            {
                "claim_id": "CLAIM_3_OPTION_B_REGRESSION_TIER_B",
                "status": claim3_status,
                "rationale": claim3_reason,
                "witnesses": ["results/science/campaign/campaign_index.csv"],
                "next_test": "Run regime ablation and bootstrap confidence intervals." if claim3_status != "ACCEPTED" else "",
            }
        )

        if "UNDERDETERMINED" in {claim2_status, claim3_status}:
            tier_b_status = "UNDERDETERMINED"
        elif "REJECTED" in {claim2_status, claim3_status}:
            tier_b_status = "REJECTED"
        else:
            tier_b_status = "ACCEPTED"

        rows.append(
            {
                "claim_id": "SUPPORTED_CLAIMS_TIER_B_SUMMARY",
                "status": tier_b_status,
                "rationale": "Tier B summary over Claim 2 and Claim 3 regression checks.",
                "witnesses": ["results/science/campaign/campaign_index.csv", "results/science/campaign/campaign_report.md"],
                "next_test": "Run missing or failed Tier B regressions to reach conclusive status." if tier_b_status == "UNDERDETERMINED" else "",
            }
        )

        rows.append(
            {
                "claim_id": "PDF_FRAMEWORK_TRACEABILITY",
                "status": "ACCEPTED" if tier_b_status in {"ACCEPTED", "REJECTED"} else "UNDERDETERMINED",
                "rationale": "PDF claims are mapped to Tier B executable regression evidence.",
                "witnesses": [str(pdf_path), "results/selection/evidence_index.json"],
            }
        )
    elif focus == "C":
        latest_by_test: dict[str, dict[str, Any]] = {}
        for row in campaign_rows:
            test_id = str(row.get("test_id", "")).strip()
            if test_id:
                latest_by_test[test_id] = row

        c_tests = ["claim3p_ising_cyclic_l8", "claim3p_heisenberg_cyclic_l8", "claim3p_l16_gate"]
        missing = [t for t in c_tests if t not in latest_by_test]

        pass_set = {"PASS", "SUPPORTED", "ACCEPTED"}
        reject_set = {"FAIL", "REJECTED"}

        if missing:
            c_status = "UNDERDETERMINED"
            c_rationale = f"Missing Tier C tests: {', '.join(missing)}."
            c_next = f"Run missing Tier C tests: {', '.join(missing)}."
        else:
            verdicts = [str(latest_by_test[t].get("verdict", "")).upper() for t in c_tests]
            if all(v in reject_set for v in verdicts):
                c_status = "REJECTED"
                c_rationale = "All Tier C Claim 3P tests failed; rejection is conclusive for current hypotheses."
                c_next = "Propose alternative hypotheses/falsifiers before additional Tier C runs."
            elif all(v in pass_set for v in verdicts):
                c_status = "ACCEPTED"
                c_rationale = "All Tier C Claim 3P tests passed; support is conclusive for current hypotheses."
                c_next = ""
            else:
                c_status = "UNDERDETERMINED"
                c_rationale = "Tier C Claim 3P evidence is mixed."
                c_next = "Generate discriminative follow-up tests from failed/passed divergence."

        row_c: dict[str, Any] = {
            "claim_id": "CLAIM_3P_PHYSICAL_CONVERGENCE_TIER_C",
            "status": c_status,
            "rationale": c_rationale,
            "witnesses": ["results/science/campaign/campaign_index.csv", "results/science/campaign/campaign_report.md"],
        }
        if c_next:
            row_c["next_test"] = c_next
        rows.append(row_c)

        rows.append(
            {
                "claim_id": "PDF_FRAMEWORK_TRACEABILITY",
                "status": "ACCEPTED" if c_status in {"ACCEPTED", "REJECTED"} else "UNDERDETERMINED",
                "rationale": "PDF physics claims are mapped to Tier C executable convergence evidence.",
                "witnesses": [str(pdf_path), "results/selection/evidence_index.json"],
            }
        )
    else:
        claim3p_status = "UNDERDETERMINED"
        if science_status == "REJECTED":
            claim3p_status = "REJECTED"
        elif science_status == "SUPPORTED":
            claim3p_status = "ACCEPTED"

        claim3p_row = {
            "claim_id": "CLAIM_3P_PHYSICAL_CONVERGENCE",
            "status": claim3p_status,
            "rationale": "Mapped from Claim 3P baseline/exploration verdict artifacts.",
            "witnesses": [
                science_artifacts.get("claim3p_baseline", "results/science/baseline_summary.json"),
                science_artifacts.get("campaign_report", "results/science/campaign/campaign_report.md"),
            ],
        }
        if claim3p_row["status"] == "UNDERDETERMINED":
            claim3p_row["next_test"] = "Run L=16 heisenberg_cyclic and ising_cyclic with >=3 seeds and compare ΔAIC/ΔBIC stability."
        rows.append(claim3p_row)

        rows.append(
            {
                "claim_id": "CLAIM_3_OPTION_B",
                "status": "UNDERDETERMINED",
                "rationale": "Option B evidence exists but physical interpretation remains limited to local numerical support.",
                "witnesses": [
                    science_artifacts.get("claim3_baseline", "results/science/baseline_summary.json"),
                    "results/science/campaign/model_comparison.json",
                ],
                "next_test": "Add independent fit-family ablation and bootstrap confidence intervals for model ranking.",
            }
        )

        # Keep at least one explicit claim tied directly to the proposal PDF.
        rows.append(
            {
                "claim_id": "PDF_FRAMEWORK_TRACEABILITY",
                "status": "UNDERDETERMINED",
                "rationale": "PDF is present and indexed, but strong external physical claims require independent external benchmarks.",
                "witnesses": [
                    str(pdf_path),
                    "results/selection/evidence_index.json",
                ],
                "next_test": "Map each physical claim sentence to one executable test and one artifact-backed falsifier.",
            }
        )

    selection_manifest = {
        "generated_at_utc": utc_now(),
        "pdf_source": str(pdf_path),
        "focus_objective": focus,
        "entry_count": len(rows),
        "statuses": ["ACCEPTED", "UNDERDETERMINED", "REJECTED"],
        "required_fields": ["claim_id", "status", "rationale", "witnesses"],
    }

    report_lines = [
        "# Selection Report",
        "",
        f"Source proposal: `{pdf_path}`",
        "",
        "| Claim | Status | Witnesses |",
        "|---|---|---:|",
    ]
    for row in rows:
        report_lines.append(f"| {row['claim_id']} | {row['status']} | {len(row['witnesses'])} |")
    report_lines.append("")
    report_lines.append("## Next Tests")
    for row in rows:
        if row.get("status") == "UNDERDETERMINED":
            report_lines.append(f"- {row['claim_id']}: {row.get('next_test', 'TBD')}")

    write_json(selection_dir / "evidence_index.json", evidence_index)
    write_json(selection_dir / "selection_manifest.json", selection_manifest)
    with (selection_dir / "ledger.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (selection_dir / "selection_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Selection pipeline health is independent from scientific claim outcomes.
    # REJECTED claims are valid outputs and must not fail Step 5 by themselves.
    if any(r["status"] == "UNDERDETERMINED" for r in rows):
        status = "UNDERDETERMINED"
    elif any(r["status"] == "REJECTED" for r in rows):
        status = "REJECTED"
    else:
        status = "ACCEPTED"

    return status, rows


def package_retention(run_dir: Path, artifacts_root: Path) -> dict[str, Any]:
    run_id = run_dir.name
    retained_dir = artifacts_root / "retained_runs"
    retained_dir.mkdir(parents=True, exist_ok=True)

    temp_archive = artifacts_root / "tmp" / f"retained_{run_id}.tar.gz"
    temp_archive.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(temp_archive, "w:gz") as tar:
        tar.add(run_dir, arcname=run_id)

    local_archive = run_dir / f"retained_{run_id}.tar.gz"
    shutil.copy2(temp_archive, local_archive)

    retained_archive = retained_dir / local_archive.name
    shutil.copy2(temp_archive, retained_archive)

    size = local_archive.stat().st_size
    retained_text = "\n".join(
        [
            f"archive_filename: {local_archive.name}",
            f"archive_size_bytes: {size}",
            f"retention_path: {retained_archive}",
            f"timestamp_utc: {utc_now()}",
            "remote_uri: N/A",
        ]
    )
    (run_dir / "RETAINED.txt").write_text(retained_text + "\n", encoding="utf-8")

    return {
        "archive_filename": local_archive.name,
        "archive_size_bytes": size,
        "retention_path": str(retained_archive),
        "timestamp_utc": utc_now(),
        "remote_uri": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WORKFLOW_AUTO orchestrator")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help="Directory where RUN_* and retained artifacts are written (defaults to external data repo when available)",
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run when --run-id points to a prior RUN_* directory.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the latest RUN_* directory in artifacts root when --run-id is not provided.",
    )
    parser.add_argument("--pdf-path", type=Path, default=None)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--max-runs", type=int, default=80)
    parser.add_argument("--local-only-max-minutes", type=int, default=20)
    parser.add_argument("--local-only-max-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-science", action="store_true")
    parser.add_argument(
        "--focus-objective",
        type=str,
        default="ALL",
        choices=["ALL", "A", "B", "C"],
        help="Critical-path objective focus for planning/selection. Use A to prioritize platform readiness.",
    )
    parser.add_argument(
        "--tier-c-justification",
        type=str,
        default="",
        help="One-line reason to allow Tier C physics exploration when RUN_OVERRIDE_TIER_C=1",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifacts_root = choose_artifacts_root(repo_root, args.artifacts_root)
    python_exe = get_python(repo_root)

    run_id = args.run_id.strip()
    if not run_id and args.resume_latest:
        latest = find_latest_run_dir(artifacts_root)
        if latest is not None:
            run_id = latest.name
    run_id = run_id or f"RUN_{compact_ts()}"
    run_dir = artifacts_root / run_id
    resume_requested = bool(args.resume or args.resume_latest)
    existing_run = run_dir.exists()
    resume_enabled = resume_requested and existing_run
    if existing_run and not resume_requested:
        raise RuntimeError(
            f"Run directory already exists: {run_dir}. Re-run with --resume (or --resume-latest) to continue."
        )

    logs_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    tmp_dir = run_dir / "tmp"

    for p in [
        logs_dir,
        results_dir,
        tmp_dir,
        results_dir / "contracts",
        results_dir / "sdk_validation",
        results_dir / "science" / "campaign",
        results_dir / "selection",
    ]:
        p.mkdir(parents=True, exist_ok=True)
    if not resume_enabled or not (results_dir / "capabilities.json").exists():
        write_json(results_dir / "capabilities.json", {"generated_at_utc": utc_now(), "status": "NOT_COLLECTED"})

    events_path = logs_dir / "workflow_events.jsonl"

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "repo_root": str(repo_root),
        "artifacts_root": str(artifacts_root),
        "started_utc": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "user": os.environ.get("USER", "unknown"),
        },
        "steps": {},
        "commands": [],
        "mcp_calls": [],
        "failure_classification": None,
        "failure_message": None,
        "mcp_targets": {
            "compute_target": None,
            "docs_target": None,
            "discovery_log": "logs/mcporter_list.txt",
        },
    }
    if resume_enabled:
        existing_manifest = safe_load_json(run_dir / "manifest.json")
        if existing_manifest is not None:
            manifest = existing_manifest
            manifest["run_id"] = run_id
            manifest["repo_root"] = str(repo_root)
            manifest["artifacts_root"] = str(artifacts_root)
            manifest.setdefault("started_utc", utc_now())
            manifest.setdefault("steps", {})
            manifest.setdefault("commands", [])
            manifest.setdefault("mcp_calls", [])
            manifest.setdefault(
                "mcp_targets",
                {
                    "compute_target": None,
                    "docs_target": None,
                    "discovery_log": "logs/mcporter_list.txt",
                },
            )
        manifest.setdefault("resume_history", []).append({"resumed_utc": utc_now()})
    manifest["_resume_enabled"] = resume_enabled

    stop_reason: str | None = None
    stop_classification: str | None = None
    lint_status = "PASS"
    contract_status = "SKIPPED"
    science_status = "NOT_RUN"
    selection_status = "NOT_RUN"
    retention_status = "NOT_RUN"
    mode = "COMPLETE"
    campaign_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    step4_executed = False
    step5_executed = False

    cgf_proc: subprocess.Popen[str] | None = None
    cgf_endpoint = "http://127.0.0.1:8080"

    status_obj = load_result_json(results_dir, "workflow_auto_status.json") if resume_enabled else None
    verdict_obj = load_result_json(results_dir, "VERDICT.json") if resume_enabled else None
    if status_obj is not None:
        contract_status = str(status_obj.get("contract_status", contract_status))
        science_status = str(status_obj.get("science_status", science_status))
        selection_status = str(status_obj.get("selection_status", selection_status))
        retention_status = str(status_obj.get("retention_status", retention_status))
        if str(status_obj.get("mode", "")).upper() == "LOCAL_ONLY":
            mode = "LOCAL_ONLY"
    if args.focus_objective == "A":
        # Tier A readiness runs are independent of physics-claim status.
        science_status = "NOT_RUN"
    if verdict_obj is not None:
        lint_status = str(verdict_obj.get("lint_status", lint_status))
    if resume_enabled:
        campaign_rows = [{} for _ in range(count_csv_rows(results_dir / "science" / "campaign" / "campaign_index.csv"))]
        ledger_rows = read_jsonl_rows(results_dir / "selection" / "ledger.jsonl")

    def can_skip_step(step_id: str, required_paths: list[str]) -> bool:
        if not resume_enabled:
            return False
        step_obj = manifest.get("steps", {}).get(step_id, {})
        if str(step_obj.get("status", "")).upper() != "PASS":
            return False
        return all((run_dir / rel).exists() for rel in required_paths)

    try:
        # Step 0: preflight and MCP discovery
        step = "0"
        if can_skip_step(step, ["logs/mcporter_list.txt", "results/capabilities.json"]):
            manifest["steps"].setdefault(step, {"name": "Preflight and MCP discovery", "status": "PASS"})
            record_event(events_path, step, "step_skipped_resume", {"status": "PASS"})
            if manifest.get("mcp_targets", {}).get("compute_target") is None:
                mode = "LOCAL_ONLY"
                args.max_minutes = args.local_only_max_minutes
                args.max_runs = args.local_only_max_runs
        else:
            manifest["steps"][step] = {"name": "Preflight and MCP discovery", "status": "RUNNING"}
            record_event(events_path, step, "step_started", {"name": manifest["steps"][step]["name"]})

            preflight_cmds = [
                ("env_uname", ["uname", "-a"]),
                ("env_python_version", [python_exe, "--version"]),
                ("env_pip_freeze", [python_exe, "-m", "pip", "freeze"]),
                ("git_remote_v", ["git", "remote", "-v"]),
                ("git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                ("git_commit", ["git", "rev-parse", "--short", "HEAD"]),
                ("git_status_porcelain", ["git", "status", "--porcelain"]),
            ]
            for name, cmd in preflight_cmds:
                run_command(
                    run_dir=run_dir,
                    manifest=manifest,
                    events_path=events_path,
                    step=step,
                    name=name,
                    command=cmd,
                    cwd=repo_root,
                )

            mcporter_cmd = run_command(
                run_dir=run_dir,
                manifest=manifest,
                events_path=events_path,
                step=step,
                name="mcporter_cmd_check",
                command=["sh", "-lc", "command -v mcporter"],
                cwd=repo_root,
            )
            if mcporter_cmd["exit_code"] != 0:
                stop_reason = "mcporter not found; cannot satisfy Step 0.4 discovery"
                stop_classification = "RUNTIME_FAILURE"
                mode = "STOPPED"
            else:
                mcporter_list = run_command(
                    run_dir=run_dir,
                    manifest=manifest,
                    events_path=events_path,
                    step=step,
                    name="mcporter_list",
                    command=["mcporter", "list"],
                    cwd=repo_root,
                )
                manifest["mcp_calls"].append(
                    {
                        "tool": "mcporter list",
                        "server_target": "local",
                        "request_summary": "discover available MCP servers",
                        "response_pointer": mcporter_list["log_path"],
                    }
                )

                mcporter_text = read_text(run_dir / mcporter_list["log_path"])
                compute_target, docs_target, discovered = parse_mcporter_targets(mcporter_text)
                manifest["mcp_targets"] = {
                    "compute_target": compute_target,
                    "docs_target": docs_target,
                    "discovery_log": "logs/mcporter_list.txt",
                    "servers": discovered,
                }

                capabilities = discover_capabilities(repo_root, discovered)
                write_json(results_dir / "capabilities.json", capabilities)
                manifest["capabilities"] = {"path": "results/capabilities.json", "skills_count": len(capabilities["skills"])}
                record_event(
                    events_path,
                    step,
                    "capability_discovery",
                    {
                        "commands_available": [k for k, v in capabilities["commands"].items() if v["available"]],
                        "skills_count": len(capabilities["skills"]),
                        "mcp_servers": discovered,
                    },
                )

                if compute_target is None:
                    mode = "LOCAL_ONLY"
                    args.max_minutes = args.local_only_max_minutes
                    args.max_runs = args.local_only_max_runs

            manifest["steps"][step]["status"] = "PASS" if stop_reason is None else "FAIL"
            record_event(events_path, step, "step_finished", {"status": manifest["steps"][step]["status"]})

            if stop_reason:
                raise RuntimeError(stop_reason)

        # Step 1: repo verification
        step = "1"
        if can_skip_step(step, []):
            manifest["steps"].setdefault(step, {"name": "Repo verification", "status": "PASS"})
            record_event(events_path, step, "step_skipped_resume", {"status": "PASS", "lint_status": lint_status})
        else:
            manifest["steps"][step] = {"name": "Repo verification", "status": "RUNNING"}
            record_event(events_path, step, "step_started", {"name": manifest["steps"][step]["name"]})

            if not args.skip_install:
                install = run_command(
                    run_dir=run_dir,
                    manifest=manifest,
                    events_path=events_path,
                    step=step,
                    name="pip_install_requirements",
                    command=[python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
                    cwd=repo_root,
                )
                if install["exit_code"] != 0:
                    stop_reason = "Dependency installation failed"
                    stop_classification = "RUNTIME_FAILURE"
                    mode = "STOPPED"
                    raise RuntimeError(stop_reason)

            if not args.skip_lint:
                lint = run_command(
                    run_dir=run_dir,
                    manifest=manifest,
                    events_path=events_path,
                    step=step,
                    name="make_lint",
                    command=["make", "lint"],
                    cwd=repo_root,
                )
                if lint["exit_code"] != 0:
                    # Safe autofix once.
                    run_command(
                        run_dir=run_dir,
                        manifest=manifest,
                        events_path=events_path,
                        step=step,
                        name="ruff_fix",
                        command=["ruff", "check", ".", "--fix"],
                        cwd=repo_root,
                    )
                    run_command(
                        run_dir=run_dir,
                        manifest=manifest,
                        events_path=events_path,
                        step=step,
                        name="ruff_format",
                        command=["ruff", "format", "."],
                        cwd=repo_root,
                    )
                    lint_rerun = run_command(
                        run_dir=run_dir,
                        manifest=manifest,
                        events_path=events_path,
                        step=step,
                        name="make_lint_rerun",
                        command=["make", "lint"],
                        cwd=repo_root,
                    )
                    if lint_rerun["exit_code"] != 0:
                        lint_status = "LINT_DEBT"

            if not args.skip_tests:
                test_env = os.environ.copy()
                test_env["ALLOW_CGF_DOWN"] = "1"
                makefile_text = read_text(repo_root / "Makefile")
                use_test_fast = bool(re.search(r"(?m)^\s*test-fast\s*:", makefile_text))
                if use_test_fast:
                    tests = run_command(
                        run_dir=run_dir,
                        manifest=manifest,
                        events_path=events_path,
                        step=step,
                        name="make_test_fast",
                        command=["make", "test-fast"],
                        cwd=repo_root,
                        env=test_env,
                    )
                else:
                    tests = run_command(
                        run_dir=run_dir,
                        manifest=manifest,
                        events_path=events_path,
                        step=step,
                        name="pytest_tests",
                        command=[python_exe, "-m", "pytest", "-q", "tests/"],
                        cwd=repo_root,
                        env=test_env,
                    )
                if tests["exit_code"] != 0:
                    stop_reason = "Step 1 tests failed"
                    stop_classification = "TEST_FAILURE"
                    mode = "STOPPED"
                    raise RuntimeError(stop_reason)

            manifest["steps"][step]["status"] = "PASS"
            record_event(events_path, step, "step_finished", {"status": "PASS", "lint_status": lint_status})

        # Step 2: CGF bring-up + contract verification
        step = "2"
        if can_skip_step(step, ["results/contracts/contract_result.json", "results/sdk_validation/sdk_validation_result.json"]):
            manifest["steps"].setdefault(step, {"name": "CGF contract verification", "status": "PASS"})
            record_event(events_path, step, "step_skipped_resume", {"status": "PASS", "contract_status": contract_status})
        else:
            manifest["steps"][step] = {"name": "CGF contract verification", "status": "RUNNING"}
            record_event(events_path, step, "step_started", {"name": manifest["steps"][step]["name"]})

            server_started = False
            for port in PORT_ROTATION:
                if not is_port_available(port):
                    continue
                log_path = logs_dir / f"cgf_server_{port}.txt"
                env = os.environ.copy()
                env["CGF_PORT"] = str(port)
                env["PYTHONUNBUFFERED"] = "1"
                with log_path.open("w", encoding="utf-8") as fout:
                    cgf_proc = subprocess.Popen(  # noqa: S603
                        [python_exe, "server/cgf_server_v03.py"],
                        cwd=str(repo_root),
                        env=env,
                        stdout=fout,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                cgf_endpoint = f"http://127.0.0.1:{port}"
                for _ in range(16):
                    if cgf_proc.poll() is not None:
                        break
                    ok, _, _ = http_health(cgf_endpoint, timeout_s=0.8)
                    if ok and cgf_proc.poll() is None:
                        server_started = True
                        break
                    time.sleep(0.5)
                if server_started:
                    break
                if cgf_proc and cgf_proc.poll() is None:
                    cgf_proc.terminate()
                    cgf_proc.wait(timeout=5)
                    cgf_proc = None

            if not server_started:
                stop_reason = "CGF server failed to start on any allowed port"
                stop_classification = "RUNTIME_FAILURE"
                mode = "STOPPED"
                raise RuntimeError(stop_reason)

            manifest["cgf"] = {"endpoint": cgf_endpoint}
            health_ok, status_code, health_body = http_health(cgf_endpoint, timeout_s=2.0)
            write_json(
                results_dir / "cgf_health.json",
                {
                    "endpoint": cgf_endpoint,
                    "ok": health_ok,
                    "status_code": status_code,
                    "body": health_body,
                    "checked_at_utc": utc_now(),
                },
            )
            if not health_ok:
                stop_reason = "CGF health check failed"
                stop_classification = "RUNTIME_FAILURE"
                mode = "STOPPED"
                raise RuntimeError(stop_reason)

            contract_cmd = [python_exe, "-m", "pytest", "-v", "tools/contract_compliance_tests.py"]
            # Keep run_contract_suite.sh as a preferred command only when we are on the default
            # endpoint; the shell wrapper currently hard-codes 8080 checks.
            if cgf_endpoint.endswith(":8080") and (repo_root / "tools/run_contract_suite.sh").exists():
                contract_cmd = ["bash", "tools/run_contract_suite.sh"]

            contract_env = os.environ.copy()
            contract_env["CGF_ENDPOINT"] = cgf_endpoint
            contract = run_command(
                run_dir=run_dir,
                manifest=manifest,
                events_path=events_path,
                step=step,
                name="contract_suite",
                command=contract_cmd,
                cwd=repo_root,
                env=contract_env,
            )

            write_json(
                results_dir / "contracts" / "contract_result.json",
                {
                    "command": contract_cmd,
                    "exit_code": contract["exit_code"],
                    "log_path": contract["log_path"],
                    "endpoint": cgf_endpoint,
                    "generated_at_utc": utc_now(),
                },
            )

            if contract["exit_code"] != 0:
                contract_status = "FAIL"
                stop_reason = "Contract suite failed"
                stop_classification = "CONTRACT_FAILURE"
                mode = "STOPPED"
                raise RuntimeError(stop_reason)

            sdk = run_command(
                run_dir=run_dir,
                manifest=manifest,
                events_path=events_path,
                step=step,
                name="validate_sdk_artifacts",
                command=[python_exe, "tools/validate_sdk_artifacts.py"],
                cwd=repo_root,
            )

            write_json(
                results_dir / "sdk_validation" / "sdk_validation_result.json",
                {
                    "command": [python_exe, "tools/validate_sdk_artifacts.py"],
                    "exit_code": sdk["exit_code"],
                    "log_path": sdk["log_path"],
                    "generated_at_utc": utc_now(),
                },
            )

            if sdk["exit_code"] != 0:
                contract_status = "FAIL"
                stop_reason = "SDK validation failed"
                stop_classification = "CONTRACT_FAILURE"
                mode = "STOPPED"
                raise RuntimeError(stop_reason)

            contract_status = "PASS" if lint_status == "PASS" else "LINT_DEBT"
            manifest["steps"][step]["status"] = "PASS"
            record_event(events_path, step, "step_finished", {"status": "PASS"})

        runners = locate_baseline_runners(repo_root)

        # Step 3: baseline science
        step = "3"
        if can_skip_step(step, ["results/science/run_plan.json", "results/science/baseline_summary.json"]):
            manifest["steps"].setdefault(step, {"name": "Scientific baseline suite", "status": "PASS"})
            record_event(events_path, step, "step_skipped_resume", {"status": "PASS", "science_status": science_status})
        else:
            manifest["steps"][step] = {"name": "Scientific baseline suite", "status": "RUNNING"}
            record_event(events_path, step, "step_started", {"name": manifest["steps"][step]["name"]})

            run_plan = {
                "generated_at_utc": utc_now(),
                "baseline_tests": [
                    {"test_id": "claim2_tradeoff_baseline", "runner": str(runners["claim2"]), "seed": 123},
                    {"test_id": "claim3_optionb_baseline", "runner": str(runners["claim3"]), "seed": args.seed},
                    {"test_id": "claim3p_ising_open_baseline", "runner": str(runners["claim3p"]), "seed": args.seed},
                ],
                "exploration_tests": [],
                "budgets": {
                    "max_minutes": args.max_minutes,
                    "max_runs": args.max_runs,
                    "max_failures": 10,
                    "seeds": 3,
                    "mode": mode,
                },
                "stop_conditions": {
                    "hard_failures": ["TEST_FAILURE", "CONTRACT_FAILURE", "RUNTIME_FAILURE", "SELECTION_FAILURE"],
                    "science_evidence_failure_is_nonblocking": True,
                },
            }
            write_json(results_dir / "science" / "run_plan.json", run_plan)

            baseline_outputs = {
                "claim2_baseline": results_dir / "science" / "claim2_baseline",
                "claim3_baseline": results_dir / "science" / "claim3_baseline",
                "claim3p_ising_open": results_dir / "science" / "claim3p_ising_open",
            }

            baseline_commands = [
                (
                    "science_claim2_baseline",
                    [python_exe, str(runners["claim2"]), "--output", str(baseline_outputs["claim2_baseline"]), "--seed", "123"],
                    "claim2_baseline",
                ),
                (
                    "science_claim3_optionb_baseline",
                    [
                        python_exe,
                        str(runners["claim3"]),
                        "--output_root",
                        str(results_dir / "science"),
                        "--experiment_name",
                        "claim3_baseline",
                        "--chi_sweep",
                        "2,4,8",
                        "--seeds_per_chi",
                        "3",
                        "--num_sites",
                        "32",
                        "--subsystem_sizes",
                        "16,8,4",
                        "--seed_base",
                        str(args.seed),
                    ],
                    "claim3_baseline",
                ),
                (
                    "science_claim3p_ising_open",
                    [
                        python_exe,
                        str(runners["claim3p"]),
                        "--L",
                        "8",
                        "--A_size",
                        "4",
                        "--model",
                        "ising_open",
                        "--chi_sweep",
                        "2,4",
                        "--restarts_per_chi",
                        "1",
                        "--fit_steps",
                        "30",
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(baseline_outputs["claim3p_ising_open"]),
                    ],
                    "claim3p_ising_open",
                ),
            ]

            baseline_summary: dict[str, Any] = {"generated_at_utc": utc_now(), "tests": []}

            if args.skip_science or args.focus_objective == "A":
                reason = "--skip-science set" if args.skip_science else "--focus-objective A suppresses science baseline"
                baseline_summary["tests"].append({"test_id": "ALL_BASELINE", "status": "NOT_RUN", "reason": reason})
                science_status = "NOT_RUN"
            else:
                for name, cmd, key in baseline_commands:
                    entry = run_command(
                        run_dir=run_dir,
                        manifest=manifest,
                        events_path=events_path,
                        step=step,
                        name=name,
                        command=cmd,
                        cwd=repo_root,
                    )

                    artifact_root = baseline_outputs[key]
                    artifact_path = find_latest_subdir(artifact_root) or artifact_root
                    verdict = detect_science_verdict(artifact_path)

                    baseline_summary["tests"].append(
                        {
                            "test_id": key,
                            "command": cmd,
                            "exit_code": entry["exit_code"],
                            "log_path": entry["log_path"],
                            "artifact_path": str(artifact_path.relative_to(run_dir)),
                            "verdict": verdict,
                        }
                    )

                verdicts = [t["verdict"] for t in baseline_summary["tests"]]
                if any(v == "REJECTED" for v in verdicts):
                    science_status = "REJECTED"
                elif any(v == "SUPPORTED" for v in verdicts):
                    science_status = "SUPPORTED"
                elif any(v == "INCONCLUSIVE" for v in verdicts):
                    science_status = "PARTIAL"
                else:
                    science_status = "NOT_RUN"

            write_json(results_dir / "science" / "baseline_summary.json", baseline_summary)
            manifest["steps"][step]["status"] = "PASS"
            record_event(events_path, step, "step_finished", {"status": "PASS", "science_status": science_status})

        # Step 4: exploration
        step = "4"
        step4_executed = True
        manifest["steps"][step] = {"name": "Exploration campaign", "status": "RUNNING"}
        record_event(events_path, step, "step_started", {"name": manifest["steps"][step]["name"]})

        proposal_json = results_dir / "science" / "exploration_proposals.json"
        proposal_md = results_dir / "science" / "exploration_proposals.md"
        planner_cmd = [
            python_exe,
            "tools/plan_framework_selection_tests.py",
            "--repo-root",
            str(repo_root),
            "--artifacts-root",
            str(run_dir),
            "--catalog",
            "docs/physics/framework_selection_test_catalog_v1.json",
            "--max-minutes",
            str(args.max_minutes),
            "--max-runs",
            str(args.max_runs),
            "--output-json",
            str(proposal_json),
            "--output-md",
            str(proposal_md),
            "--focus-objective",
            args.focus_objective,
        ]
        compute_target = str(manifest.get("mcp_targets", {}).get("compute_target") or "").strip()
        if compute_target:
            planner_cmd.extend(["--compute-target", compute_target])
        if mode == "LOCAL_ONLY":
            planner_cmd.append("--local-only")
        if args.tier_c_justification.strip():
            planner_cmd.extend(["--tier-c-justification", args.tier_c_justification.strip()])

        planner = run_command(
            run_dir=run_dir,
            manifest=manifest,
            events_path=events_path,
            step=step,
            name="plan_framework_selection_tests",
            command=planner_cmd,
            cwd=repo_root,
        )
        if planner["exit_code"] != 0:
            stop_reason = "Framework-selection planner failed"
            stop_classification = "RUNTIME_FAILURE"
            mode = "STOPPED"
            raise RuntimeError(stop_reason)

        plan_obj = json.loads(proposal_json.read_text(encoding="utf-8"))
        proposals = plan_obj.get("proposals", [])
        selected = plan_obj.get("selected", [])
        remaining_minutes = int(plan_obj.get("remaining_minutes", args.max_minutes))

        write_json(
            results_dir / "science" / "exploration_selected.json",
            {
                "generated_at_utc": utc_now(),
                "budget_start_minutes": args.max_minutes,
                "budget_remaining_minutes": remaining_minutes,
                "selected": selected,
                "stop_condition": plan_obj.get("stop_condition", {}),
            },
        )

        campaign_rows: list[dict[str, Any]] = []
        model_comparison: dict[str, Any] = {"generated_at_utc": utc_now(), "tests": []}
        skipped_tests: list[str] = []

        if not args.skip_science:
            for test in selected:
                test_id = test.get("execution_key") or test["test_id"]
                chi_label = "2,4"
                seed_used = str(args.seed)
                if test_id == "claim3p_ising_cyclic_l8":
                    output_root = results_dir / "science" / "claim3p_cyclic_L8"
                    cmd = [
                        python_exe,
                        str(runners["claim3p"]),
                        "--L",
                        "8",
                        "--A_size",
                        "4",
                        "--model",
                        "ising_cyclic",
                        "--chi_sweep",
                        "2,4",
                        "--restarts_per_chi",
                        "1",
                        "--fit_steps",
                        "30",
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(output_root),
                    ]
                    model_name = "ising_cyclic"
                    L = 8
                elif test_id == "claim3p_heisenberg_cyclic_l8":
                    output_root = results_dir / "science" / "claim3p_heisenberg_cyclic_L8"
                    cmd = [
                        python_exe,
                        str(runners["claim3p"]),
                        "--L",
                        "8",
                        "--A_size",
                        "4",
                        "--model",
                        "heisenberg_cyclic",
                        "--chi_sweep",
                        "2,4",
                        "--restarts_per_chi",
                        "1",
                        "--fit_steps",
                        "30",
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(output_root),
                    ]
                    model_name = "heisenberg_cyclic"
                    L = 8
                elif test_id == "claim3_optionb_regime_check":
                    output_root = results_dir / "science"
                    cmd = [
                        python_exe,
                        str(runners["claim3"]),
                        "--output_root",
                        str(output_root),
                        "--experiment_name",
                        "claim3_explore",
                        "--chi_sweep",
                        "2,4,8",
                        "--seeds_per_chi",
                        "3",
                        "--num_sites",
                        "32",
                        "--subsystem_sizes",
                        "16,8,4",
                        "--seed_base",
                        str(args.seed + 7),
                    ]
                    model_name = "optionB"
                    L = 32
                    chi_label = "2,4,8"
                    seed_used = str(args.seed + 7)
                elif test_id == "claim2_seed_perturbation":
                    output_root = results_dir / "science" / "claim2_seed321"
                    cmd = [python_exe, str(runners["claim2"]), "--output", str(output_root), "--seed", "321"]
                    model_name = "claim2"
                    L = 0
                    chi_label = "n/a"
                    seed_used = "321"
                elif test_id == "claim3p_l16_gate":
                    output_root = results_dir / "science" / "claim3p_cyclic_L16"
                    cmd = [
                        python_exe,
                        str(runners["claim3p"]),
                        "--L",
                        "16",
                        "--A_size",
                        "8",
                        "--model",
                        "ising_cyclic",
                        "--chi_sweep",
                        "2,4,8",
                        "--restarts_per_chi",
                        "1",
                        "--fit_steps",
                        "40",
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(output_root),
                    ]
                    model_name = "ising_cyclic"
                    L = 16
                    chi_label = "2,4,8"
                elif test_id == "platform_openclaw_opt_check":
                    output_root = results_dir / "science" / "platform_openclaw_opt_check"
                    output_root.mkdir(parents=True, exist_ok=True)
                    cmd = [python_exe, "tools/openclaw_opt_check.py", "--output", str(output_root / "openclaw_opt_check.json")]
                    model_name = "platform"
                    L = 0
                    chi_label = "n/a"
                    seed_used = "n/a"
                elif test_id == "platform_openclaw_opt_check_relaxed":
                    output_root = results_dir / "science" / "platform_openclaw_opt_check_relaxed"
                    output_root.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        python_exe,
                        "tools/openclaw_opt_check.py",
                        "--output",
                        str(output_root / "openclaw_opt_check.json"),
                        "--min-files",
                        "5",
                        "--min-chunks",
                        "20",
                        "--min-eligible-skills",
                        "8",
                        "--max-disabled-skills",
                        "6",
                    ]
                    model_name = "platform"
                    L = 0
                    chi_label = "n/a"
                    seed_used = "n/a"
                elif test_id == "platform_mcporter_health_snapshot":
                    output_root = results_dir / "science" / "platform_mcporter_health_snapshot"
                    output_root.mkdir(parents=True, exist_ok=True)
                    cmd = ["mcporter", "list"]
                    model_name = "platform"
                    L = 0
                    chi_label = "n/a"
                    seed_used = "n/a"
                else:
                    skipped_tests.append(str(test_id))
                    continue

                c_entry = run_command(
                    run_dir=run_dir,
                    manifest=manifest,
                    events_path=events_path,
                    step=step,
                    name=f"explore_{test_id}",
                    command=cmd,
                    cwd=repo_root,
                )
                if test_id == "platform_mcporter_health_snapshot":
                    (output_root / "mcporter_list.txt").write_text(
                        read_text(run_dir / c_entry["log_path"]),
                        encoding="utf-8",
                    )
                artifact_path = find_latest_subdir(output_root) or output_root
                verdict_path = artifact_path / "verdict.json"
                verdict = "NOT_RUN"
                delta_aic = None
                delta_bic = None
                key_metrics: dict[str, Any] = {}
                if verdict_path.exists():
                    verdict_obj = json.loads(verdict_path.read_text(encoding="utf-8"))
                    verdict = str(verdict_obj.get("verdict", "NOT_RUN"))
                    falsifiers = verdict_obj.get("falsifiers", {})
                    p34 = falsifiers.get("P3.4") or falsifiers.get("3.3_model_selection") or {}
                    delta_aic = p34.get("delta_aic")
                    delta_bic = p34.get("delta_bic")
                    key_metrics = verdict_obj.get("metrics", {}) if isinstance(verdict_obj.get("metrics"), dict) else {}
                elif test_id.startswith("platform_openclaw_opt_check"):
                    opt_path = output_root / "openclaw_opt_check.json"
                    if opt_path.exists():
                        opt = safe_load_json(opt_path) or {}
                        ready = bool(opt.get("ready", False))
                        verdict = "PASS" if ready else "FAIL"
                        key_metrics = {
                            "ready": ready,
                            "score": opt.get("score"),
                            "errors": opt.get("errors"),
                            "warnings": opt.get("warnings"),
                            "finding_codes": [f.get("code") for f in opt.get("findings", []) if isinstance(f, dict)],
                        }
                    else:
                        verdict = "PASS" if c_entry["exit_code"] == 0 else "FAIL"
                elif c_entry["exit_code"] == 0:
                    verdict = "PASS"
                else:
                    verdict = "FAIL"

                campaign_rows.append(
                    {
                        "timestamp": utc_now(),
                        "test_id": test_id,
                        "model": model_name,
                        "L": L,
                        "chi": chi_label,
                        "seed": seed_used,
                        "key_metrics": json.dumps(key_metrics),
                        "aic_delta": "" if delta_aic is None else delta_aic,
                        "bic_delta": "" if delta_bic is None else delta_bic,
                        "verdict": verdict,
                        "artifact_path": str(artifact_path.relative_to(run_dir)),
                        "exit_code": c_entry["exit_code"],
                    }
                )
                model_comparison["tests"].append(
                    {
                        "test_id": test_id,
                        "verdict": verdict,
                        "delta_aic": delta_aic,
                        "delta_bic": delta_bic,
                        "artifact_path": str(artifact_path.relative_to(run_dir)),
                    }
                )

        campaign_index_path = results_dir / "science" / "campaign" / "campaign_index.csv"
        with campaign_index_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "test_id",
                    "model",
                    "L",
                    "chi",
                    "seed",
                    "key_metrics",
                    "aic_delta",
                    "bic_delta",
                    "verdict",
                    "artifact_path",
                    "exit_code",
                ],
            )
            writer.writeheader()
            for row in campaign_rows:
                writer.writerow(row)

        report_lines = [
            "# Campaign Report",
            "",
            f"Generated: {utc_now()}",
            f"Mode: {mode}",
            "",
            "| Test | Verdict | AIC delta | BIC delta | Artifact |",
            "|---|---|---:|---:|---|",
        ]
        for row in campaign_rows:
            report_lines.append(
                f"| {row['test_id']} | {row['verdict']} | {row['aic_delta']} | {row['bic_delta']} | `{row['artifact_path']}` |"
            )
        if not campaign_rows:
            report_lines.append("| none | NOT_RUN |  |  |  |")
        if skipped_tests:
            report_lines.append("")
            report_lines.append("## Skipped Planned Tests")
            for test_id in skipped_tests:
                report_lines.append(f"- {test_id}")

        (results_dir / "science" / "campaign" / "campaign_report.md").write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )
        write_json(results_dir / "science" / "campaign" / "model_comparison.json", model_comparison)
        manifest["steps"][step]["status"] = "PASS"
        record_event(events_path, step, "step_finished", {"status": "PASS", "rows": len(campaign_rows)})

        # Step 5: selection
        step = "5"
        if can_skip_step(
            step,
            [
                "results/selection/evidence_index.json",
                "results/selection/selection_manifest.json",
                "results/selection/ledger.jsonl",
                "results/selection/selection_report.md",
            ],
        ) and not step4_executed:
            manifest["steps"].setdefault(step, {"name": "Selection pass", "status": "PASS"})
            ledger_rows = read_jsonl_rows(results_dir / "selection" / "ledger.jsonl")
            record_event(
                events_path,
                step,
                "step_skipped_resume",
                {"status": "PASS", "selection_status": selection_status, "entries": len(ledger_rows)},
            )
        else:
            step5_executed = True
            manifest["steps"][step] = {"name": "Selection pass", "status": "RUNNING"}
            record_event(events_path, step, "step_started", {"name": manifest["steps"][step]["name"]})

            pdf_path = find_pdf(repo_root, args.pdf_path)
            if pdf_path is None:
                stop_reason = "Framework with Selection PDF not found in discovery paths"
                stop_classification = "SELECTION_FAILURE"
                mode = "STOPPED"
                raise RuntimeError(stop_reason)

            evidence_index = {
                "generated_at_utc": utc_now(),
                "pdf_source": str(pdf_path),
                "claims": {
                    "CGR_CONTRACT_COMPLIANCE": {
                        "artifacts": [
                            "results/contracts/contract_result.json",
                            "results/sdk_validation/sdk_validation_result.json",
                        ],
                        "descriptor": "Governance and SDK contract status",
                    },
                    "CLAIM_2": {
                        "artifacts": [
                            "results/science/claim2_baseline",
                            "results/science/baseline_summary.json",
                        ],
                        "descriptor": "Capacity tradeoff baseline evidence",
                    },
                    "CLAIM_3": {
                        "artifacts": [
                            "results/science/claim3_baseline",
                            "results/science/campaign/campaign_index.csv",
                        ],
                        "descriptor": "Entropy scaling and regime checks",
                    },
                    "CLAIM_3P": {
                        "artifacts": [
                            "results/science/claim3p_ising_open",
                            "results/science/campaign/model_comparison.json",
                        ],
                        "descriptor": "Physical convergence checks",
                    },
                },
            }

            science_artifacts = {
                "claim3p_baseline": "results/science/claim3p_ising_open",
                "claim3_baseline": "results/science/claim3_baseline",
                "campaign_report": "results/science/campaign/campaign_report.md",
            }

            selection_status, ledger_rows = build_selection_outputs(
                run_dir=run_dir,
                pdf_path=pdf_path,
                evidence_index=evidence_index,
                contract_status=contract_status,
                science_status=science_status,
                science_artifacts=science_artifacts,
                campaign_rows=campaign_rows,
                focus_objective=args.focus_objective,
            )

            manifest["steps"][step]["status"] = "PASS" if selection_status != "FAIL" else "FAIL"
            record_event(
                events_path,
                step,
                "step_finished",
                {
                    "status": manifest["steps"][step]["status"],
                    "selection_status": selection_status,
                    "entries": len(ledger_rows),
                },
            )

            if selection_status == "FAIL":
                stop_reason = "Selection status was FAIL"
                stop_classification = "SELECTION_FAILURE"
                mode = "STOPPED"
                raise RuntimeError(stop_reason)

        # Step 6: final verdict, summary, retention
        step = "6"
        if can_skip_step(
            step,
            [
                "results/workflow_auto_status.json",
                "results/VERDICT.json",
                "results/index.json",
                "summary.md",
                "RETAINED.txt",
            ],
        ) and not step4_executed and not step5_executed:
            manifest["steps"].setdefault(step, {"name": "Verdict and retention", "status": "PASS"})
            retention_status = "PASS"
            record_event(events_path, step, "step_skipped_resume", {"status": "PASS"})
        else:
            manifest["steps"][step] = {"name": "Verdict and retention", "status": "RUNNING"}
            record_event(events_path, step, "step_started", {"name": manifest["steps"][step]["name"]})

            overall_status = "COMPLETE"
            focus = (args.focus_objective or "ALL").strip().upper()
            if mode == "STOPPED":
                overall_status = "STOPPED"
            elif focus in {"A", "B", "C"}:
                # Focused objective runs are conclusive once selection is ACCEPTED/REJECTED.
                if selection_status in {"UNDERDETERMINED", "NOT_RUN", "FAIL"}:
                    overall_status = "PARTIAL"
            elif selection_status in {"UNDERDETERMINED"} or science_status in {"PARTIAL", "INCONCLUSIVE", "NOT_RUN", "REJECTED"}:
                overall_status = "PARTIAL"

            workflow_status = {
                "run_id": run_id,
                "mode": mode,
                "mcp_targets": {
                    "compute_target": manifest["mcp_targets"].get("compute_target"),
                    "docs_target": manifest["mcp_targets"].get("docs_target"),
                    "discovery_log": "logs/mcporter_list.txt",
                },
                "contract_status": contract_status,
                "science_status": science_status,
                "selection_status": selection_status,
                "retention_status": "NOT_RUN",
                "artifact_index": "results/index.json",
                "notes": [
                    f"lint_status={lint_status}",
                    f"focus_objective={args.focus_objective}",
                    f"overall_status={overall_status}",
                ],
            }

            verdict = {
                "overall_status": overall_status,
                "lint_status": lint_status,
                "contract_status": contract_status,
                "scientific_status": {
                    "status": science_status,
                    "baseline_summary": "results/science/baseline_summary.json",
                    "campaign_report": "results/science/campaign/campaign_report.md",
                },
                "selection_status": selection_status,
                "deltas_vs_baseline_expectations": {
                    "local_only_mode": mode == "LOCAL_ONLY",
                    "exploration_runs": len(campaign_rows),
                },
                "key_artifact_pointers": {
                    "contracts": "results/contracts",
                    "sdk_validation": "results/sdk_validation",
                    "science": "results/science",
                    "selection": "results/selection",
                    "capabilities": "results/capabilities.json",
                    "workflow_status": "results/workflow_auto_status.json",
                },
                "generated_at_utc": utc_now(),
            }

            index_obj = {
                "run_id": run_id,
                "generated_at_utc": utc_now(),
                "pointers": {
                    "contracts": {"path": "results/contracts"},
                    "sdk_validation": {"path": "results/sdk_validation"},
                    "science": {"path": "results/science"},
                    "selection": {"path": "results/selection"},
                    "capabilities": {"path": "results/capabilities.json"},
                    "VERDICT": {"path": "results/VERDICT.json"},
                    "VERDICT_latest": {"path": "results/VERDICT_latest.json"},
                    "workflow_status": {"path": "results/workflow_auto_status.json"},
                    "workflow_status_latest": {"path": "results/workflow_auto_status_latest.json"},
                    "result_history": {"path": "results/history/history_index.jsonl"},
                },
            }
            write_json(results_dir / "index.json", index_obj)

            summary_lines = [
                "# WORKFLOW_AUTO Summary",
                "",
                f"Run: `{run_id}`",
                f"Mode: `{mode}`",
                "",
                "## 1. What ran",
                "- Baseline suite: claim2/claim3/claim3p (local deterministic commands)",
                "- Exploration subset executed under budget constraints",
                "",
                "## 2. What was learned and what changed",
                f"- Contract status: `{contract_status}`",
                f"- Science status: `{science_status}`",
                f"- Selection status: `{selection_status}`",
                "",
                "## 3. Current conclusions",
                "- Evidence is retained under `results/` and selected claims are classified in `results/selection/ledger.jsonl`.",
                "",
                "## 4. Underdetermined items and next tests",
                "- See underdetermined entries in `results/selection/ledger.jsonl` with `next_test` recommendations.",
                "",
                "## 5. Troubleshooting and fixes applied",
                f"- Port rotation used for CGF startup: {PORT_ROTATION}",
                "- LOCAL_ONLY downscoping auto-applied when compute MCP target is unavailable.",
                "",
                "## 6. Rerun commands",
                f"- `python3 tools/run_workflow_auto.py --repo-root {repo_root} --artifacts-root {artifacts_root} --run-id {run_id} --resume`",
                f"- `python3 tools/validate_workflow_auto_run.py --run-dir {run_dir}`",
            ]
            (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

            retention = package_retention(run_dir, artifacts_root)
            retention_status = "PASS"
            workflow_status["retention_status"] = retention_status
            write_result_with_history(
                results_dir=results_dir,
                filename="workflow_auto_status.json",
                obj=workflow_status,
                freeze_legacy=True,
            )
            write_result_with_history(
                results_dir=results_dir,
                filename="VERDICT.json",
                obj=verdict,
                freeze_legacy=True,
            )

            manifest["steps"][step]["status"] = "PASS"
            record_event(events_path, step, "step_finished", {"status": "PASS", "retention": retention})

    except Exception as exc:  # noqa: BLE001
        if stop_classification is None:
            stop_classification = "RUNTIME_FAILURE"
        if stop_classification not in FAILURE_CODES:
            stop_classification = "RUNTIME_FAILURE"

        if stop_reason is None:
            stop_reason = str(exc)

        manifest["failure_classification"] = stop_classification
        manifest["failure_message"] = stop_reason

        if mode != "STOPPED":
            mode = "STOPPED"

        # Always emit minimum required files for post-mortem.
        fallback_status = {
            "run_id": run_id,
            "mode": mode,
            "mcp_targets": {
                "compute_target": manifest["mcp_targets"].get("compute_target"),
                "docs_target": manifest["mcp_targets"].get("docs_target"),
                "discovery_log": "logs/mcporter_list.txt",
            },
            "contract_status": contract_status,
            "science_status": science_status,
            "selection_status": selection_status,
            "retention_status": retention_status,
            "artifact_index": "results/index.json",
            "notes": [f"failure_classification={stop_classification}", f"failure_message={stop_reason}"],
        }
        write_result_with_history(
            results_dir=results_dir,
            filename="workflow_auto_status.json",
            obj=fallback_status,
            freeze_legacy=True,
        )

        write_result_with_history(
            results_dir=results_dir,
            filename="VERDICT.json",
            obj={
                "overall_status": "STOPPED",
                "lint_status": lint_status,
                "contract_status": contract_status,
                "scientific_status": {"status": science_status},
                "selection_status": selection_status,
                "error": {"classification": stop_classification, "message": stop_reason},
                "generated_at_utc": utc_now(),
            },
            freeze_legacy=True,
        )

        if not (results_dir / "index.json").exists():
            write_json(
                results_dir / "index.json",
                {
                    "run_id": run_id,
                    "generated_at_utc": utc_now(),
                    "pointers": {
                        "contracts": {"path": "results/contracts"},
                        "sdk_validation": {"path": "results/sdk_validation"},
                        "science": {"path": "results/science"},
                        "selection": {"path": "results/selection"},
                        "capabilities": {"path": "results/capabilities.json"},
                        "VERDICT": {"path": "results/VERDICT.json"},
                        "VERDICT_latest": {"path": "results/VERDICT_latest.json"},
                        "workflow_status": {"path": "results/workflow_auto_status.json"},
                        "workflow_status_latest": {"path": "results/workflow_auto_status_latest.json"},
                        "result_history": {"path": "results/history/history_index.jsonl"},
                    },
                },
            )

        if not (run_dir / "summary.md").exists():
            (run_dir / "summary.md").write_text(
                "# WORKFLOW_AUTO Summary\n\n"
                f"Run: `{run_id}`\n\n"
                f"Status: STOPPED ({stop_classification})\n\n"
                f"Reason: {stop_reason}\n",
                encoding="utf-8",
            )

    finally:
        if cgf_proc and cgf_proc.poll() is None:
            cgf_proc.terminate()
            try:
                cgf_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cgf_proc.kill()

        manifest["ended_utc"] = utc_now()
        write_json(run_dir / "manifest.json", manifest)

    # Validate resulting run artifacts once for immediate feedback.
    validator_cmd = [python_exe, "tools/validate_workflow_auto_run.py", "--run-dir", str(run_dir), "--output", str(results_dir / "workflow_contract_validation.json")]
    subprocess.run(validator_cmd, cwd=str(repo_root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    print(str(run_dir))
    return 0 if mode != "STOPPED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
