# Developer Guide - Host Adapters

## Python Version

Requires Python 3.10+

## Node Version (for JS hook)

Requires Node.js 18+ (for the OpenClaw ES module hook)

## How to Run CGF Server

```bash
# Start the CGF server
python server/cgf_server_v03.py

# Server runs on http://127.0.0.1:8080 by default
```

## How to Run Compliance Tests

```bash
# With CGF running
pytest -v tools/contract_compliance_tests.py

# Without CGF (test runner handles "0 tests")
./tools/run_contract_suite.sh
```

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
