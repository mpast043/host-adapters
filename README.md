# Host Adapters — Capacity Governance Framework (CGF)

CGF host adapters for OpenClaw, LangGraph, and other AI agent runtimes.
The framework intercepts tool calls and memory writes, routes them through
a deterministic policy engine, and enforces ALLOW / BLOCK / CONSTRAIN / AUDIT
decisions with a full audit trail.

## Repository Structure

```
.
├── sdk/                          # CGF Python SDK (v0.1.0)
│   └── python/cgf_sdk/
│       ├── __init__.py
│       ├── adapter_base.py       # Abstract HostAdapter base class
│       ├── cgf_client.py         # Typed REST client (async + sync)
│       └── errors.py             # Canonical exception hierarchy
├── adapters/                     # Host-specific adapter implementations
│   ├── openclaw_adapter_v02.py   # OpenClaw (schema 0.2/0.3, current)
│   ├── openclaw_adapter_v01.py   # OpenClaw legacy (schema 0.1)
│   ├── langgraph_adapter_v01.py  # LangGraph (schema 0.3, current)
│   └── openclaw_cgf_hook_v02.mjs # JS ES-module hook for OpenClaw
├── server/                       # CGF decision server (FastAPI)
│   ├── cgf_server_v03.py         # Server v0.3 with policy engine
│   ├── cgf_schemas_v03.py        # Schema v0.3 type definitions
│   └── cgf_schemas_v02.py        # Schema v0.2 (backward compat)
├── cgf_policy/                   # Policy Engine v1.0 (deterministic)
│   ├── compiler.py               # Bundle loader & hash validator
│   ├── evaluator.py              # Rule evaluation engine
│   ├── fields.py                 # Safe field accessors
│   └── types.py                  # Pydantic type definitions
├── policy/                       # Policy configuration files
│   └── policy_bundle_v1.json     # Default policy bundle (6 rules)
├── tests/                        # Test suite
│   ├── test_policy_engine.py     # Policy engine unit tests (16 tests)
│   └── test_outcome_reporting.py # Outcome reporting / audit trail tests
├── tools/                        # Validation & CI tools
│   ├── contract_compliance_tests.py
│   ├── run_contract_suite.sh     # Full CI gate (starts CGF, runs tests, lints)
│   ├── schema_lint.py
│   └── replay_verify.py
├── experiments/                  # MERA/CGF research (separate from governance code)
├── DEV.md                        # Developer guide
├── requirements.txt              # Python dependencies
└── Makefile                      # Build automation
```

## Quick Start

```bash
# 1. Install dependencies (includes httpx + requests for all code paths)
pip install -r requirements.txt

# 2. Run the full test gate
make test           # policy engine tests + contract compliance suite

# 3. Quick iteration (no CGF server required)
make test-fast      # policy engine tests only
```

## Running the CGF Server

```bash
# Default port 8080
python3 server/cgf_server_v03.py

# With policy bundle (recommended)
CGF_POLICY_BUNDLE_PATH=policy/policy_bundle_v1.json python3 server/cgf_server_v03.py

# Custom port
CGF_PORT=8082 python3 server/cgf_server_v03.py
```

**Health check**: `GET /v1/health` (returns 200 with JSON status)

## Exception Hierarchy (SDK canonical — v0.5.1+)

All adapters now raise exceptions from `sdk/python/cgf_sdk/errors.py`.
Callers can catch the same exceptions regardless of which adapter is in use:

| Exception | Meaning |
|-----------|---------|
| `GovernanceError` | Base class for all governance errors |
| `ActionBlockedError` | Action blocked by policy (BLOCK decision) |
| `ActionConstrainedError` | Constraint failed to apply |
| `FailModeError` | CGF unreachable, fail mode applied (defer) |
| `CGFConnectionError` | Network / timeout failure reaching CGF |
| `CGFRegistryError` | Adapter registration failed |

```python
from cgf_sdk.errors import ActionBlockedError, GovernanceError

try:
    await adapter.governance_hook_tool("file_write", args, ...)
except ActionBlockedError as e:
    print(f"Blocked [{e.error_code}]: {e}")   # works for both OpenClaw & LangGraph
except GovernanceError as e:
    print(f"Governance issue: {e}")
```

## Strict Policy Mode

Set `CGF_STRICT=1` on the server to change the default `ALLOW` outcome for
unknown tools to `AUDIT`. Useful in production environments where unknown
tools should be flagged rather than silently permitted.

```bash
CGF_STRICT=1 python3 server/cgf_server_v03.py
```

See [DEV.md](DEV.md) for full documentation including env vars, test commands,
and fail-mode configuration.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CGF_ENDPOINT` | `http://127.0.0.1:8080` | CGF server URL |
| `CGF_TIMEOUT_MS` | `500` | Request timeout ms |
| `CGF_DATA_DIR` | `./cgf_data/` | Local data directory |
| `CGF_PORT` | `8080` | CGF server listen port |
| `CGF_POLICY_BUNDLE_PATH` | `policy/policy_bundle_v1.json` | Policy bundle path |
| `CGF_STRICT` | `0` | Strict mode: `1` → AUDIT unknown tools |

## Governance Lifecycle

```
1. OBSERVE  — adapter extracts proposal, context, signals from host
2. EVALUATE — CGF policy engine returns ALLOW/BLOCK/CONSTRAIN/AUDIT/DEFER
3. ENFORCE  — adapter applies decision locally
4. REPORT   — outcome sent to CGF (local JSONL fallback; never silent)
```

Every step emits a canonical `HostEvent` (19 types) written to a local JSONL
file for offline audit and replay verification.

## Schema Version

This repository uses **Schema v0.3.0** with backward compatibility to v0.2.x.

## Experiments

The `experiments/` directory contains MERA tensor-network physics simulations
(Claim 3P, Claim 3A) unrelated to the governance framework. See
[CONSOLIDATED_REPORT_CLAIM3P.md](CONSOLIDATED_REPORT_CLAIM3P.md) for results.
⚠️ See Section 6 (ERRATA) for a critical Hamiltonian convention bug affecting L=8 results.

## License

MIT
