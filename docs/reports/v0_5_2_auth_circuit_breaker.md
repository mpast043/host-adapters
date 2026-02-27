# Patch Summary — v0.5.2

## Overview

This patch adds two opt-in reliability and security features to the CGF host-adapter stack.
No public API shapes were changed; all changes are additive or internal.
All existing tests continue to pass; 29 new tests were added.

---

## What Changed

### Feature A — Optional Bearer-Token Auth
**Files**: `server/cgf_server_v03.py`, `tests/test_server_auth.py`

- New env var `CGF_AUTH_TOKEN` (default `""`). When empty, auth is disabled — all requests
  pass and behavior is identical to v0.5.1.
- When set, `POST /v1/register`, `POST /v1/evaluate`, and `POST /v1/outcomes/report` require:
  `Authorization: Bearer <token>`
- Token comparison uses `hmac.compare_digest()` (constant-time, resistant to timing attacks).
- `GET /v1/health` is always unprotected.
- Missing or wrong token → HTTP 401 Unauthorized.
- 13 new tests in `tests/test_server_auth.py` using FastAPI `TestClient`.

### Feature B — Circuit Breaker
**Files**: `sdk/python/cgf_sdk/cgf_client.py`, `sdk/python/cgf_sdk/errors.py`,
`tests/test_circuit_breaker.py`

- New `CircuitBreaker` class in `cgf_client.py` with three states:
  - **CLOSED** — normal operation; consecutive failures increment counter.
  - **OPEN** — CGF is considered down; calls raise `CGFConnectionError(error_code="CIRCUIT_OPEN")`
    immediately (no network call). Transitions to HALF_OPEN after `CGF_CB_COOLDOWN_MS`.
  - **HALF_OPEN** — one probe call is allowed; success → CLOSED (reset), failure → OPEN.
- Per-`CGFClient`-instance (not global) so adapters are independent.
- Controlled by env vars (all optional, all have safe defaults):

  | Var | Default | Effect |
  |-----|---------|--------|
  | `CGF_CIRCUIT_BREAKER` | `0` | `1` → enable circuit breaker |
  | `CGF_CB_FAILURE_THRESHOLD` | `3` | Failures before OPEN |
  | `CGF_CB_COOLDOWN_MS` | `2000` | ms before HALF_OPEN |
  | `CGF_CB_HALF_OPEN_MAX_CALLS` | `1` | Probes allowed in HALF_OPEN |

- Disabled by default (`CGF_CIRCUIT_BREAKER=0`) — existing behavior unchanged.
- Applied to all six call paths: `register_async`, `evaluate_async`, `report_outcome_async`,
  and their three synchronous (`requests`) equivalents.
- `errors.py`: `CGFConnectionError.__init__` now accepts an optional `error_code` parameter
  (default `"CGF_UNREACHABLE"`) so circuit-open errors can be distinguished as `"CIRCUIT_OPEN"`.
  This is a backward-compatible change — all existing callsites are unaffected.
- Both `openclaw_adapter_v02.py` and `langgraph_adapter_v01.py` catch `CGFConnectionError`
  generically, so `CIRCUIT_OPEN` is handled by the existing fail-mode table with zero adapter
  code changes.
- 16 new tests in `tests/test_circuit_breaker.py` covering all state transitions and
  CGFClient integration.

---

## New Env Vars / Behaviors

| Var | Default | Effect |
|-----|---------|--------|
| `CGF_AUTH_TOKEN` | `""` | Bearer token for write endpoints (empty = disabled) |
| `CGF_CIRCUIT_BREAKER` | `0` | `1` → enable circuit breaker in CGF client |
| `CGF_CB_FAILURE_THRESHOLD` | `3` | Failures before OPEN |
| `CGF_CB_COOLDOWN_MS` | `2000` | ms before HALF_OPEN |
| `CGF_CB_HALF_OPEN_MAX_CALLS` | `1` | Probes allowed in HALF_OPEN |

---

## How to Run Tests

```bash
# Quick (no CGF server needed, ~2s)
make test-fast
python3 -m pytest -q tests/

# Full gate
make test

# Specific new suites
python3 -m pytest tests/test_server_auth.py -v
python3 -m pytest tests/test_circuit_breaker.py -v
```

---

## Test Counts

| Suite | Before | After |
|-------|--------|-------|
| Policy engine (`test_policy_engine.py`) | 16 passed | 16 passed |
| Outcome reporting (`test_outcome_reporting.py`) | 4 passed | 4 passed |
| Server auth (`test_server_auth.py`) | — (new) | 13 passed |
| Circuit breaker (`test_circuit_breaker.py`) | — (new) | 16 passed |
| Contract compliance | 8 passed | 8 passed |
| **Total pytest** | 20 | **49** |

---

## Files Changed

| File | Change |
|------|--------|
| `server/cgf_server_v03.py` | `CGF_AUTH_TOKEN`, `require_auth()`, `Depends` on 3 endpoints |
| `sdk/python/cgf_sdk/cgf_client.py` | `CircuitBreaker` class, CB integration in all call paths |
| `sdk/python/cgf_sdk/errors.py` | Optional `error_code` param on `CGFConnectionError` |
| `tests/test_server_auth.py` | New file (13 tests) |
| `tests/test_circuit_breaker.py` | New file (16 tests) |
| `DEV.md` | Auth, circuit breaker, and updated env var table |
| `README.md` | Auth and circuit breaker sections, updated env var table |
| `docs/reports/v0_5_2_auth_circuit_breaker.md` | This file |
