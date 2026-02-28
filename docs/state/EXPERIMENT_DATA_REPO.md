# Experimental Data Repository

Heavy generated artifacts are separated from this code repository.

## Canonical Data Repo
- Local: `/tmp/openclaws/Repos/host-adapters-experimental-data`
- GitHub: `https://github.com/mpast043/host-adapters-experimental-data`

## What Lives There
- `host-adapters/RUN_*`
- `host-adapters/retained_runs/`
- `host-adapters/outputs/physics_audit/`
- `host-adapters/docs/state/physics_audit_logs/`
- `host-adapters/docs/state/evidence_index_2026-02-27.json`
- `host-adapters/docs/state/physics_audit_2026-02-27.json`
- `host-adapters/docs/state/validation_RUN_*.json`
- `host-adapters/workflow_audit_latest.json`
- `capacity-demo/outputs/framework_validation_b/` snapshot used by Tier A audit

## What Stays Here
- Governance/runtime code (`server/`, `adapters/`, `tools/`)
- Validation schemas and enforcement logic
- Pointer docs only (`docs/state/EXPERIMENT_DATA_REPO.md`, templates, product docs)

## Sync Policy
1. Generate data with tools in this repo.
2. Write artifacts directly to the data repo:
   - `make workflow-auto`
   - `make workflow-audit`
   - `python3 tools/run_physics_audit.py ...`
3. Keep code repo free of generated run/audit artifacts.
