# Developer Guide - Host Adapters

## Python Version

Requires Python 3.10+

## Node Version (for JS hook)

Requires Node.js 18+ (for the OpenClaw ES module hook)

## Virtual Environment Setup

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Also needed for tests:
pip install httpx
```

## How to Run CGF Server

```bash
# With venv activated, start the CGF server
# Default port is 8080; use CGF_PORT to override
python server/cgf_server_v03.py

# Or with custom port
CGF_PORT=8082 python server/cgf_server_v03.py

# Server runs on http://127.0.0.1:8080 by default (or CGF_PORT)
```

**Health Endpoint**: `/v1/health` (returns 200)

**Note**: The canonical health endpoint is `/v1/health`. `/health` (without /v1/) returns 404.

## How to Run Compliance Tests (Strict Mode)

The contract gate requires both CGF_PORT and CGF_ENDPOINT to be set if not using default port 8080.

```bash
# With CGF running on default port 8080
pytest -v tools/contract_compliance_tests.py

# With CGF on custom port (e.g., 8082)
CGF_PORT=8082 CGF_ENDPOINT=http://127.0.0.1:8082 ./tools/run_contract_suite.sh

# Without CGF running (test runner handles "0 tests")
./tools/run_contract_suite.sh
```

**Acceptance Criteria (strict mode)**: 8 passed, 0 failed, schema lint files > 0, events > 0, errors = 0

## How to Run Schema Lint

```bash
# Validate JSONL files
python tools/schema_lint.py --dir ./outputs/

# Strict mode (fails on warnings)
python tools/schema_lint.py --dir ./outputs/ --strict
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CGF_ENDPOINT` | `http://127.0.0.1:8080` | CGF server URL |
| `CGF_TIMEOUT_MS` | `500` | Request timeout ms |
| `CGF_DATA_DIR` | `./cgf_data/` | Local data directory |
| `CGF_PORT` | `8080` | CGF server listen port |
| `CGF_POLICY_BUNDLE_PATH` | `policy/policy_bundle_v1.json` | Policy bundle to load |
| `CGF_STRICT` | `0` | Strict mode — set to `1` to audit unknown tools |
| `CGF_AUTH_TOKEN` | `""` | Bearer token for write endpoints (empty = disabled) |
| `CGF_CIRCUIT_BREAKER` | `0` | `1` → enable circuit breaker in CGF client |
| `CGF_CB_FAILURE_THRESHOLD` | `3` | Failures before circuit opens |
| `CGF_CB_COOLDOWN_MS` | `2000` | ms before circuit transitions to HALF_OPEN |
| `CGF_CB_HALF_OPEN_MAX_CALLS` | `1` | Probe calls allowed in HALF_OPEN state |

## Bearer Token Auth (`CGF_AUTH_TOKEN`)

By default the server has no authentication. Set `CGF_AUTH_TOKEN` to require a bearer
token on all write endpoints (`POST /v1/register`, `POST /v1/evaluate`,
`POST /v1/outcomes/report`). `GET /v1/health` is always unprotected.

```bash
CGF_AUTH_TOKEN=mysecret python3 server/cgf_server_v03.py

# Correct token → passes auth (may get 422 for bad body)
curl -X POST http://127.0.0.1:8080/v1/evaluate \
  -H "Authorization: Bearer mysecret" \
  -H "Content-Type: application/json" -d '{}'

# Missing or wrong token → 401 Unauthorized
curl -X POST http://127.0.0.1:8080/v1/evaluate -H "Content-Type: application/json" -d '{}'
```

> **Note**: The contract suite runs without `CGF_AUTH_TOKEN` set (auth disabled), so all
> 8 tests pass unchanged.

## Circuit Breaker (`CGF_CIRCUIT_BREAKER=1`)

By default the CGF client retries until timeout on every call even when the server is down.
Set `CGF_CIRCUIT_BREAKER=1` on the **adapter** process to enable a three-state circuit breaker
that short-circuits calls after repeated failures:

- **CLOSED** → normal; failures count toward threshold.
- **OPEN** → calls raise `CGFConnectionError(error_code="CIRCUIT_OPEN")` immediately.
- **HALF_OPEN** → one probe call allowed after cooldown; success → CLOSED, failure → OPEN.

```bash
CGF_CIRCUIT_BREAKER=1 CGF_CB_FAILURE_THRESHOLD=3 CGF_CB_COOLDOWN_MS=2000 \
  python3 -c "from cgf_sdk.cgf_client import CGFClient; print('CB enabled')"
```

Adapters already catch `CGFConnectionError` generically, so `CIRCUIT_OPEN` is handled
by the existing fail-mode table without any adapter code changes.

## Strict Policy Mode (`CGF_STRICT=1`)

By default the policy engine has a catch-all `default-allow` rule: tools that
do not match any explicit rule are **allowed** (safe for development/testing).

Setting `CGF_STRICT=1` changes this: any tool that falls through to the
default rule receives an **AUDIT** decision instead of ALLOW. Adapters that
don't explicitly handle AUDIT treat it as allow-with-logging, so it is
non-breaking but surfaces unknown tools in governance logs.

```bash
# Run server with strict mode enabled
CGF_STRICT=1 python3 server/cgf_server_v03.py

# Run contract suite without strict mode (default, all tests pass)
./tools/run_contract_suite.sh
```

> **Note**: The contract suite itself runs without `CGF_STRICT=1`. The `ls`
> scenario (expected ALLOW) may receive AUDIT under strict mode because `ls`
> does not match the `readonly-allowlist` regex pattern.

## How to Run All Tests

```bash
# Full gate: policy engine + contract compliance suite (recommended for CI)
make test

# Quick iteration: policy engine only, no CGF server required
make test-fast

# Directly
python3 -m pytest -q tests/
./tools/run_contract_suite.sh
```

## Fail Mode Table

Adapters receive a `fail_mode_table` from `/v1/register` that maps:
- `(action_type, risk_tier)` → `fail_mode` (fail_closed, fail_open, defer)

This table is cached during registration and used when CGF is unreachable.

## Version Compatibility

| Adapter | Schema Version | CGF Server | Status |
|---------|---------------|------------|--------|
| openclaw_v02.py | 0.3.0 | v0.3 | ✅ Current |
| langgraph_v01.py | 0.3.0 | v0.3 | ✅ Current |
| openclaw_v01.py | 0.1.0 | v0.3 | ⚠️ Legacy |

---

# P8 Policy Engine v1.0

Deterministic, explainable policy evaluation for CGF.

## Quick Start

```bash
# Run CGF with policy bundle
CGF_POLICY_BUNDLE_PATH=policy/policy_bundle_v1.json python3 server/cgf_server_v03.py

# Compute/set bundle hash
python3 -c "from cgf_policy import load_policy_bundle; load_policy_bundle('policy/policy_bundle_v1.json')"

# Replay verification
 python3 tools/replay_verify.py --replaypack cgf_data/replay_xxx.json --policy policy/policy_bundle_v1.json -v
```

## Policy Bundle Structure

```json
{
  "policy_version": "1.0.0",
  "bundle_hash": "sha256...",
  "rules": [
    {
      "id": "rule-id",
      "priority": 100,
      "when": [
        {"field": "proposal.tool_name", "op": "in", "value": [...]}
      ],
      "decision": "BLOCK",
      "confidence": 1.0
    }
  ],
  "fail_modes": [...]
}
```

## Allowed Fields

Only these fields can be referenced in rules:
- `proposal.action_type` - tool_call, memory_write, etc.
- `proposal.tool_name` - Tool identifier
- `proposal.size_bytes` - Memory operation size
- `proposal.sensitivity_hint` - low/medium/high
- `proposal.risk_tier` - low/medium/high
- `proposal.estimated_cost.tokens` - Token cost
- `signals.token_rate_60s` - Throughput signal
- `signals.error_rate` - Error rate (0-1)
- `signals.avg_latency_ms` - Latency signal

## Deterministic Matching

1. Sort rules by `(-priority, id)`
2. First matching rule wins
3. Same priority: lexicographic `id` wins
4. Missing fields = condition non-match (safe default)

## Replay Verification

```bash
# Verify replay matches current policy
python3 tools/replay_verify.py --replaypack path/to/replay.json --policy policy/policy_bundle_v1.json

# Exit codes: 0 = match, 1 = mismatch
```

## Testing

```bash
pytest tests/test_policy_engine.py -v
```
