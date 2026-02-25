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
| CGF_ENDPOINT | http://127.0.0.1:8080 | CGF server URL |
| CGF_TIMEOUT_MS | 500 | Request timeout ms |
| CGF_DATA_DIR | ./cgf_data/ | Local data directory |

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
