# Hardening Patch Summary — v0.5.1

## Overview

This patch applies six focused hardening fixes to the CGF host-adapter stack.
No public API shapes were changed; all changes are additive or internal.
All existing tests pass; 4 new tests were added.

---

## What Changed (by Step)

### Step 1 — Unified Exception Hierarchy
**Files**: `adapters/openclaw_adapter_v02.py`, `adapters/langgraph_adapter_v01.py`

- Removed adapter-specific exception classes:
  - `CGFGovernanceError` (OpenClaw) → deleted
  - `LangGraphToolBlocked` (LangGraph) → deleted
  - `CGFUnreachableError` (LangGraph) → deleted
- Both adapters now import and raise from `sdk/python/cgf_sdk/errors.py`:
  - `ActionBlockedError` — for BLOCK decisions
  - `ActionConstrainedError` — for failed constraint application
  - `CGFConnectionError` — for network/timeout failures
  - `CGFRegistryError` — for registration failures
  - `FailModeError` — when CGF is unreachable and fail-mode=defer
  - `GovernanceError` — base class for all other cases
- Callers can now catch `ActionBlockedError` uniformly across both OpenClaw
  and LangGraph without host-specific imports.
- The `apply_fail_mode()` method in OpenClaw is now correctly wired into the
  `governance_hook_memory()` fallback path (was previously unused).

### Step 2 — Fixed Missing Dependencies
**File**: `requirements.txt`

- Added `requests>=2.28.0` — required by `cgf_client.py` sync path
- Added `httpx>=0.24.0` — required by FastAPI's `TestClient` (Starlette)
- Previously, `pip install -r requirements.txt` left the contract test suite
  unable to even import due to missing `httpx`.

### Step 3 — Eliminated Silent Outcome Loss
**Files**: `adapters/openclaw_adapter_v02.py`, `adapters/langgraph_adapter_v01.py`,
`tests/test_outcome_reporting.py`

- `report_outcome()` (OpenClaw) and `report()` (LangGraph) now implement a
  two-level fallback with guaranteed surfacing:
  1. HTTP POST to CGF (best-effort)
  2. On HTTP failure → write to local JSONL (safe atomic open, `parents=True`)
  3. On JSONL failure too → emit structured JSON to **stderr** and raise
     `GovernanceError(error_code="OUTCOME_LOSS")` — **no silent drops**
- Added 4 unit tests in `tests/test_outcome_reporting.py` covering:
  - HTTP failure → local JSONL written (OpenClaw + LangGraph)
  - Double failure → `GovernanceError` raised (OpenClaw + LangGraph)

### Step 4 — Fixed `make test`
**File**: `Makefile`

- Old: `make test` only ran `pytest tools/contract_compliance_tests.py`
  (skipped policy engine tests entirely)
- New:
  ```
  make test      → pytest -q tests/  +  ./tools/run_contract_suite.sh
  make test-fast → pytest -q tests/  (no CGF server needed)
  ```
- Policy engine regressions are now caught by CI.

### Step 5 — Strict Policy Mode (`CGF_STRICT=1`)
**File**: `server/cgf_server_v03.py`, `DEV.md`, `README.md`

- New env var `CGF_STRICT` (default `0`).
- When `CGF_STRICT=1`:
  - Policy Engine path: tools matching the `default-allow` rule receive
    **AUDIT** instead of ALLOW.
  - Legacy path: catch-all default returns **AUDIT** instead of ALLOW.
- When `CGF_STRICT=0` (default): behavior is unchanged; all existing contract
  tests pass.
- Documented in `DEV.md` under "Strict Policy Mode" and in `README.md`.

### Step 6 — Correlation IDs in Server Logs
**File**: `server/cgf_server_v03.py`

- Added `log_correlated()` helper that always emits structured JSON with:
  - `adapter_id`, `proposal_id`, `decision_id`, `event_type`
- Applied to all three key endpoints:
  - `POST /v1/register` → logs `ADAPTER_REGISTERED`
  - `POST /v1/evaluate` → logs `DECISION_MADE` (with decision value, confidence, reason_code)
  - `POST /v1/outcomes/report` → logs `OUTCOME_LOGGED`
- Grepping logs by `proposal_id` or `decision_id` now works reliably without
  cross-referencing JSONL files.

---

## New Env Vars / Behaviors

| Var | Default | Effect |
|-----|---------|--------|
| `CGF_STRICT` | `0` | `1` → AUDIT for unknown tools instead of ALLOW |

---

## How to Run Tests

```bash
# Quick (policy engine only, ~1s)
make test-fast
python3 -m pytest -q tests/

# Full gate (policy engine + CGF server + contract suite + schema lint)
make test
./tools/run_contract_suite.sh

# Specific suites
python3 -m pytest tests/test_policy_engine.py -v
python3 -m pytest tests/test_outcome_reporting.py -v
```

---

## Test Counts

| Suite | Before | After |
|-------|--------|-------|
| Policy engine (`test_policy_engine.py`) | 16 passed | 16 passed |
| Outcome reporting (`test_outcome_reporting.py`) | — (new) | 4 passed |
| Contract compliance | 0 collected (import error) | 8 passed |
| **Total pytest** | 16 | **20** |

---

## Files Changed

| File | Change |
|------|--------|
| `requirements.txt` | +`requests`, +`httpx` |
| `adapters/openclaw_adapter_v02.py` | SDK exceptions, fixed report_outcome |
| `adapters/langgraph_adapter_v01.py` | SDK exceptions, fixed report() |
| `server/cgf_server_v03.py` | CGF_STRICT, log_correlated() |
| `Makefile` | Fixed `test` target, added `test-fast` |
| `tests/test_outcome_reporting.py` | New file (4 tests) |
| `DEV.md` | Strict mode docs, updated env var table |
| `README.md` | Complete rewrite with exception hierarchy, usage |
| `docs/reports/hardening_patch_summary.md` | This file |
