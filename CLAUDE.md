# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Host Adapters - Capacity Governance Framework (CGF)

CGF host adapters for OpenClaw, LangGraph, and other AI agent runtimes.

## Project Overview

This repository implements **host adapters** that intercept tool calls and memory writes from AI agent frameworks, route them through a deterministic policy engine (CGF), and enforce governance decisions (ALLOW/BLOCK/CONSTRAIN/AUDIT/DEFER) with full audit trails.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Agent Framework│     │  Host Adapter   │     │     CGF Server   │
│ (OpenClaw,      │────▶│  (this repo)    │────▶│  (policy engine) │
│  LangGraph,     │     │                 │     │                  │
│  etc.)          │     └─────────────────┘     └──────────────────┘
└─────────────────┘           │                         │
                              │                         │
                    ┌─────────▼─────────┐     ┌─────────▼──────────┐
                    │   Local JSONL     │     │  Deterministic     │
                    │   Audit Events    │     │  Policy Rules      │
                    └───────────────────┘     └────────────────────┘
```

## Key Concepts

### Governance Lifecycle

1. **OBSERVE** - Adapter extracts proposal, context, signals from host
2. **EVALUATE** - CGF policy engine returns ALLOW/BLOCK/CONSTRAIN/AUDIT/DEFER
3. **ENFORCE** - Adapter applies decision locally
4. **REPORT** - Outcome sent to CGF (local JSONL fallback; never silent)

### Decision Types

| Decision | Behavior |
|----------|----------|
| `ALLOW` | Action permitted |
| `BLOCK` | Action blocked (raises exception) |
| `CONSTRAIN` | Action allowed with constraints applied |
| `AUDIT` | Action allowed, logged for review |
| `DEFER` | Decision postponed (usually to user) |

### Fail Modes

When CGF is unreachable, adapters apply fail modes based on `(action_type, risk_tier)`:

| Fail Mode | Behavior |
|-----------|----------|
| `fail_closed` | Block action |
| `fail_open` | Allow action |
| `defer` | Postpone decision |

## Project Structure

```
.
├── sdk/                          # CGF Python SDK
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
│   ├── test_outcome_reporting.py # Outcome reporting / audit trail tests
│   └── test_entanglement_utils.py # Physics entanglement utilities
├── tools/                        # Validation & CI tools
│   ├── contract_compliance_tests.py
│   ├── run_contract_suite.sh     # Full CI gate
│   ├── schema_lint.py
│   ├── replay_verify.py
│   └── physics/                  # Physics-specific validation tools
├── experiments/                  # Physics/MERA experiments
│   ├── claim3/                   # Claim 3P/3A/3B physics tests (see experiments/README.md)
│   ├── physics/                  # Entanglement calculations and utilities
│   ├── selection/                # Selection gate infrastructure
│   ├── truth/                    # Truth label generation and validation
│   └── verification/             # Step 3 verification engine
├── DEV.md                        # Developer guide (detailed setup)
├── requirements.txt              # Python dependencies
└── Makefile                      # Build automation
```

## Development Commands

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### Running Tests

```bash
# Full gate: policy engine + contract compliance suite (recommended for CI)
make test

# Quick iteration: policy engine only, no CGF server required
make test-fast

# Direct pytest
python3 -m pytest -q tests/
```

### Running the CGF Server

```bash
# Default port 8080
python3 server/cgf_server_v03.py

# With policy bundle (recommended)
CGF_POLICY_BUNDLE_PATH=policy/policy_bundle_v1.json python3 server/cgf_server_v03.py

# Custom port
CGF_PORT=8082 python3 server/cgf_server_v03.py
```

**Health check**: `GET /v1/health` (returns 200 with JSON status)

Note: The canonical health endpoint is `/v1/health`. `/health` (without /v1/) returns 404.

### Linting and Formatting

```bash
# Run linters
make lint

# Format code
make format

# Format JS code
make format-js
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CGF_ENDPOINT` | `http://127.0.0.1:8080` | CGF server URL |
| `CGF_TIMEOUT_MS` | `500` | Request timeout in ms |
| `CGF_DATA_DIR` | `./cgf_data/` | Local data directory |
| `CGF_PORT` | `8080` | CGF server listen port |
| `CGF_POLICY_BUNDLE_PATH` | `policy/policy_bundle_v1.json` | Policy bundle path |
| `CGF_STRICT` | `0` | Strict mode: `1` → AUDIT unknown tools |
| `CGF_AUTH_TOKEN` | `""` | Bearer token for write endpoints (empty = disabled) |
| `CGF_CIRCUIT_BREAKER` | `0` | `1` → enable circuit breaker in CGF client |
| `CGF_CB_FAILURE_THRESHOLD` | `3` | Failures before circuit opens |
| `CGF_CB_COOLDOWN_MS` | `2000` | ms before HALF_OPEN transition |
| `CGF_CB_HALF_OPEN_MAX_CALLS` | `1` | Probes allowed in HALF_OPEN |

## Schema Version

This repository uses **Schema v0.3.0** with backward compatibility to v0.2.x.

| Adapter | Schema Version | Status |
|---------|---------------|--------|
| openclaw_adapter_v02.py | 0.3.0 | ✅ Current |
| langgraph_adapter_v01.py | 0.3.0 | ✅ Current |
| openclaw_adapter_v01.py | 0.1.0 | ⚠️ Legacy |

## Exception Hierarchy (SDK)

All adapters raise exceptions from `sdk/python/cgf_sdk/errors.py`:

| Exception | Meaning |
|-----------|---------|
| `GovernanceError` | Base class for all governance errors |
| `ActionBlockedError` | Action blocked by policy (BLOCK decision) |
| `ActionConstrainedError` | Constraint failed to apply |
| `FailModeError` | CGF unreachable, fail mode applied |
| `CGFConnectionError` | Network / timeout failure |
| `CGFRegistryError` | Adapter registration failed |

## Policy Engine

The policy engine evaluates rules in priority order:

1. Rules sorted by `(-priority, id)`
2. First matching rule wins
3. Same priority: lexicographic `id` wins
4. Missing fields = condition non-match (safe default)

### Rule Structure

```json
{
  "id": "rule-id",
  "priority": 100,
  "when": [
    {"field": "proposal.tool_name", "op": "in", "value": ["exec", "bash"]}
  ],
  "decision": "BLOCK",
  "confidence": 1.0
}
```

### Allowed Rule Fields

- `proposal.action_type` - tool_call, memory_write, etc.
- `proposal.tool_name` - Tool identifier
- `proposal.size_bytes` - Memory operation size
- `proposal.sensitivity_hint` - low/medium/high
- `proposal.risk_tier` - low/medium/high
- `proposal.estimated_cost.tokens` - Token cost
- `signals.token_rate_60s` - Throughput signal
- `signals.error_rate` - Error rate (0-1)
- `signals.avg_latency_ms` - Latency signal

## Policy Bundle

Default policy bundle (`policy/policy_bundle_v1.json`) contains 6 rules:

1. **workflow-auto-exec-audit** - AUDIT workflow_auto_exec tool
2. **memory-note-write-audit** - AUDIT memory writes (under 12k tokens)
3. **small-exec-audit** - AUDIT exec tool (under 12k tokens)
4. **runtime-tool-allowlist** - ALLOW common runtime tools
5. **denylisted-tool** - BLOCK dangerous tools (file_write, exec, shell, etc.)
6. **default-allow** - ALLOW everything else (can be changed with CGF_STRICT=1)

## Architectural Principles

1. **Deterministic Policy** - All decisions made by explicit rules, no ML
2. **Eventual Consistency** - Local JSONL events written before CGF acknowledgment
3. **Fail Gracefully** - Circuit breaker prevents cascading failures when CGF is down
4. **Audit First** - All governance events persisted before decision applied
5. **Adapter Neutrality** - SDK abstracts away host-specific details
6. **Schema Versioning** - Explicit schema versioning for compatibility

## Testing Strategy

### Governance Tests

- **Unit tests** (`tests/test_policy_engine.py`) - Policy evaluation logic
- **Contract tests** (`tools/contract_compliance_tests.py`) - Full integration with CGF server

Run with: `make test` (includes both)

### Physics/MERA Tests

The `experiments/` directory contains MERA tensor-network physics simulations:

| Claim | Description | Status | Key Script |
|-------|-------------|--------|------------|
| 3A | Entanglement-max saturation | SUPPORTED | `exp3_claim3_entanglement_max_mincut_runner.py` |
| 3B | Windowed regime detection | REJECTED | `exp3_claim3_entanglement_max_mincut_runner.py` |
| 3P | Physical Hamiltonian convergence | PARTIAL | `exp3_claim3_physical_convergence_runner_v2.py` |

**Important**: There is a critical Hamiltonian convention bug discovered 2026-02-25. Always use commit `9720dfa` or later for consistent results.

```bash
# Run physics experiments (requires additional dependencies)
pip install -r experiments/requirements-exp3.txt

# Claim 3P: Physical Hamiltonian Convergence
python experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py \
    --model ising_open --L 8 --A 4 --chi 2 3 4 6 8 12 16 --steps 120 --restarts 3

# Claim 3A/3B: Entanglement-max Min-cut
python experiments/claim3/exp3_claim3_entanglement_max_mincut_runner.py \
    --L 16 --chi 2 4 6 8
```

**Verification Gate (Step 3)**:
```bash
# Run selection gates using truth infrastructure
python experiments/verification/run_step3.py \
    --claims docs/physics/claims.txt \
    --science-dir <run_dir>/results/science \
    --out <run_dir>/results/selection
```

See `experiments/README.md` and `CONSOLIDATED_REPORT_CLAIM3P.md` for full details.

## Key Files to Understand

| File | Purpose |
|------|---------|
| `sdk/python/cgf_sdk/adapter_base.py` | Base HostAdapter class with event emission |
| `sdk/python/cgf_sdk/cgf_client.py` | Typed REST client with circuit breaker |
| `cgf_policy/evaluator.py` | Rule evaluation engine |
| `server/cgf_server_v03.py` | FastAPI server with policy engine |
| `policy/policy_bundle_v1.json` | Default policy configuration |
| `adapters/openclaw_adapter_v02.py` | Current OpenClaw adapter implementation |

---

## Physics/MERA Tests

### Overview

The `experiments/` directory contains MERA (Multiscale Entanglement Renormalization Ansatz) tensor-network simulations for validating the Capacity Governance Framework against known physics results.

### Key Files

| File | Purpose |
|------|---------|
| `experiments/physics/entanglement_utils.py` | Core entanglement calculations (von Neumann, Renyi, capacity) |
| `experiments/physics/scaling_dimensions_runner.py` | d_s extraction via tensor RG |
| `experiments/physics/entanglement_gap_analysis.py` | Δλ analysis via entanglement gap measurements |
| `experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py` | MERA optimization vs ED ground states |
| `experiments/claim3/exp3_claim3_entanglement_max_mincut_runner.py` | Min-cut capacity analysis |

### Entanglement Utilities

The `experiments/physics/entanglement_utils.py` module provides:

| Function | Description |
|----------|-------------|
| `von_neumann_entropy(rho)` | S = -Tr(ρ log ρ) |
| `renyi_entropy(rho, alpha)` | S_α = (1/(1-α)) Tr(ρ^α) |
| `reduced_density_matrix(psi, subsystem_A)` | Partial trace via quimb |
| `entanglement_spectrum(rho)` | Eigenvalues of ρ_A |
| `entanglement_gap(rho)` | λ₀ - λ₁ (gap to first excited) |
| `capacity_of_entanglement(rho)` | C_E = Var(ln ρ) - second cumulant |
| `capacity_from_entanglement(S)` | C = S / S_max normalization |

### Truth Infrastructure

**Step 3 Selection Gate** validates results against known truths:

```
truth/
├── schema.py              # TruthLabel dataclass
├── truth_generator.py     # Plugin-based label generation
├── plugins/               # Claim-specific generators
│   ├── observer_claims.py   # W02-W20 analytical truths
│   └── regression_claims.py # Claim2/Claim3 physics truths
└── truth_labels/          # Committed truth JSONs
```

**Run verification**:
```bash
python experiments/verification/run_step3.py --claims <list> --science-dir <dir> --out <dir>
```

### Known Issues

**Hamiltonian Convention Bug (2026-02-25)**:

| Commit | ED Method | Ising L=8 E₀ |
|--------|-----------|--------------|
| `ac4432a` | Dense manual builder | -9.84 (WRONG) |
| `9720dfa` | Sparse quimb | -4.22 (CORRECT) |

Always use commit `9720dfa` or later. Re-run L=8 results with corrected Hamiltonian.

### Key Results Summary

| Claim | Result | Meaning |
|-------|--------|---------|
| 3A | SUPPORTED | Entanglement saturates at max (min-cut) |
| 3B | REJECTED | Windowed regime detection fails |
| 3P | PARTIAL | MERA finds ground state but model selection fails for L=8 |
