#!/usr/bin/env python3
"""Local compute MCP server for host-adapters workflows.

Provides a constrained local execution surface for experiment and test commands.
Artifacts and logs are written to the experimental-data repository by default.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


SERVER_NAME = "host-adapters-local-compute"
DEFAULT_REPO_ROOT = Path("/tmp/openclaws/Repos/host-adapters")
DEFAULT_DATA_ROOT = Path("/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters")

REPO_ROOT = Path(os.environ.get("LOCAL_COMPUTE_REPO_ROOT", str(DEFAULT_REPO_ROOT))).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("HOST_ADAPTERS_EXPERIMENTAL_DATA_DIR", str(DEFAULT_DATA_ROOT))).expanduser().resolve()
JOBS_DIR = DATA_ROOT / "openclaw_adapter_data" / "compute_jobs"
LOGS_DIR = JOBS_DIR / "logs"
JOBS_INDEX = JOBS_DIR / "jobs_index.json"

ALLOWED_BINS = {"python", "python3", "pytest", "make"}
ALLOWED_MAKE_TARGETS = {
    "test",
    "test-fast",
    "lint",
    "format",
    "workflow-auto",
    "workflow-audit",
    "openclaw-opt-check",
}


def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_INDEX.exists():
        JOBS_INDEX.write_text("[]\n", encoding="utf-8")


def now_ts() -> float:
    return time.time()


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_cwd(cwd: str | None) -> Path:
    if cwd is None or cwd.strip() == "":
        target = REPO_ROOT
    else:
        p = Path(cwd).expanduser()
        target = p if p.is_absolute() else (REPO_ROOT / p)
    target = target.resolve()
    if not (is_subpath(target, REPO_ROOT) or is_subpath(target, DATA_ROOT)):
        raise ValueError(f"cwd must be under {REPO_ROOT} or {DATA_ROOT}, got {target}")
    return target


def normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        raise ValueError("argv is required")
    cmd = str(argv[0]).strip()
    if not cmd:
        raise ValueError("argv[0] must be a command")
    base = Path(cmd).name
    if base not in ALLOWED_BINS:
        raise ValueError(f"command '{cmd}' not allowed; allowed: {sorted(ALLOWED_BINS)}")
    if base == "make" and len(argv) >= 2:
        target = str(argv[1]).strip()
        if target and not target.startswith("-") and target not in ALLOWED_MAKE_TARGETS:
            raise ValueError(f"make target '{target}' not allowed; allowed: {sorted(ALLOWED_MAKE_TARGETS)}")
    return [str(x) for x in argv]


def tail_text(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-max(1, lines) :])


def load_index() -> list[dict[str, Any]]:
    ensure_dirs()
    try:
        return json.loads(JOBS_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_index(rows: list[dict[str, Any]]) -> None:
    ensure_dirs()
    JOBS_INDEX.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def update_job(job_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    rows = load_index()
    for i, row in enumerate(rows):
        if row.get("job_id") == job_id:
            rows[i] = {**row, **patch}
            save_index(rows)
            return rows[i]
    raise ValueError(f"job_id not found: {job_id}")


def create_job(argv: list[str], cwd: Path, mode: str) -> dict[str, Any]:
    ensure_dirs()
    jid = f"job_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    log_path = LOGS_DIR / f"{jid}.log"
    row: dict[str, Any] = {
        "job_id": jid,
        "created_at_utc": utc_iso(),
        "mode": mode,
        "argv": argv,
        "cwd": str(cwd),
        "status": "RUNNING",
        "log_path": str(log_path),
        "pid": None,
        "exit_code": None,
        "duration_s": None,
        "started_ts": now_ts(),
        "ended_at_utc": None,
        "exit_path": str(LOGS_DIR / f"{jid}.exit"),
    }
    rows = load_index()
    rows.append(row)
    save_index(rows)
    return row


def refresh_job_status(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") != "RUNNING":
        return row
    exit_path = Path(str(row.get("exit_path", "")))
    if exit_path.exists():
        try:
            exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        except ValueError:
            exit_code = 1
        ended = now_ts()
        return update_job(
            str(row["job_id"]),
            {
                "status": "SUCCEEDED" if exit_code == 0 else "FAILED",
                "exit_code": exit_code,
                "duration_s": round(max(0.0, ended - float(row.get("started_ts", ended))), 3),
                "ended_at_utc": utc_iso(),
            },
        )
    pid = row.get("pid")
    if pid:
        try:
            os.kill(int(pid), 0)
        except OSError:
            return update_job(
                str(row["job_id"]),
                {"status": "UNKNOWN", "ended_at_utc": utc_iso(), "duration_s": None},
            )
    return row


mcp = FastMCP(SERVER_NAME, instructions="Constrained local compute tools for host-adapters workflows.")


@mcp.tool(description="Health and configuration details for this local compute MCP server.")
def compute_health() -> dict[str, Any]:
    ensure_dirs()
    return {
        "status": "ok",
        "server": SERVER_NAME,
        "repo_root": str(REPO_ROOT),
        "data_root": str(DATA_ROOT),
        "jobs_dir": str(JOBS_DIR),
        "allowed_bins": sorted(ALLOWED_BINS),
        "allowed_make_targets": sorted(ALLOWED_MAKE_TARGETS),
        "timestamp_utc": utc_iso(),
    }


@mcp.tool(description="Execute a command synchronously with constrained binaries and capture logs.")
def compute_exec(argv: list[str], cwd: str | None = None, timeout_seconds: int = 1800) -> dict[str, Any]:
    safe_argv = normalize_argv(argv)
    target_cwd = resolve_cwd(cwd)
    job = create_job(safe_argv, target_cwd, mode="sync")
    log_path = Path(job["log_path"])
    timeout_seconds = max(1, min(int(timeout_seconds), 8 * 3600))
    started = now_ts()

    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.run(
            safe_argv,
            cwd=str(target_cwd),
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    ended = now_ts()
    status = "SUCCEEDED" if proc.returncode == 0 else "FAILED"
    updated = update_job(
        str(job["job_id"]),
        {
            "status": status,
            "exit_code": proc.returncode,
            "duration_s": round(max(0.0, ended - started), 3),
            "ended_at_utc": utc_iso(),
        },
    )
    return {
        "job_id": updated["job_id"],
        "status": updated["status"],
        "exit_code": updated["exit_code"],
        "duration_s": updated["duration_s"],
        "cwd": updated["cwd"],
        "argv": updated["argv"],
        "log_path": updated["log_path"],
        "log_tail": tail_text(log_path, lines=120),
    }


@mcp.tool(description="Start a background command and return a job id for later polling.")
def compute_exec_background(argv: list[str], cwd: str | None = None) -> dict[str, Any]:
    safe_argv = normalize_argv(argv)
    target_cwd = resolve_cwd(cwd)
    job = create_job(safe_argv, target_cwd, mode="background")
    log_path = Path(job["log_path"])
    exit_path = Path(job["exit_path"])
    shell_cmd = f"{shlex.join(safe_argv)} >> {shlex.quote(str(log_path))} 2>&1; echo $? > {shlex.quote(str(exit_path))}"
    proc = subprocess.Popen(  # noqa: S603
        ["/bin/sh", "-lc", shell_cmd],
        cwd=str(target_cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    updated = update_job(str(job["job_id"]), {"pid": proc.pid})
    return {
        "job_id": updated["job_id"],
        "status": updated["status"],
        "pid": updated["pid"],
        "cwd": updated["cwd"],
        "argv": updated["argv"],
        "log_path": updated["log_path"],
    }


@mcp.tool(description="Get status for a compute job id.")
def compute_job_status(job_id: str) -> dict[str, Any]:
    rows = load_index()
    for row in rows:
        if str(row.get("job_id")) == str(job_id):
            updated = refresh_job_status(row)
            return updated
    raise ValueError(f"job_id not found: {job_id}")


@mcp.tool(description="Return the tail of a compute job log file.")
def compute_job_tail(job_id: str, lines: int = 200) -> dict[str, Any]:
    row = compute_job_status(job_id)
    log_path = Path(str(row["log_path"]))
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "exit_code": row.get("exit_code"),
        "log_path": str(log_path),
        "tail": tail_text(log_path, lines=max(1, min(int(lines), 2000))),
    }


@mcp.tool(description="List recent compute jobs.")
def compute_list_jobs(limit: int = 20) -> dict[str, Any]:
    rows = load_index()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(refresh_job_status(row))
    normalized.sort(key=lambda x: str(x.get("created_at_utc", "")), reverse=True)
    lim = max(1, min(int(limit), 200))
    return {"count": len(normalized), "jobs": normalized[:lim]}


@mcp.tool(description="Terminate a background job by PID if still running.")
def compute_cancel_job(job_id: str) -> dict[str, Any]:
    row = compute_job_status(job_id)
    pid = row.get("pid")
    if not pid:
        return {"job_id": job_id, "status": row.get("status"), "message": "job has no PID (likely sync)"}
    if row.get("status") != "RUNNING":
        return {"job_id": job_id, "status": row.get("status"), "message": "job already finished"}
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError as exc:
        return {"job_id": job_id, "status": "UNKNOWN", "message": f"failed to terminate pid={pid}: {exc}"}
    updated = update_job(job_id, {"status": "CANCELLED", "ended_at_utc": utc_iso()})
    return {"job_id": updated["job_id"], "status": updated["status"], "pid": pid}


if __name__ == "__main__":
    ensure_dirs()
    mcp.run(transport="stdio")
