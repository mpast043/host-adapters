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
- `capacity-demo/outputs/framework_validation_b/` snapshot used by Tier A audit

## What Stays Here
- Governance/runtime code (`server/`, `adapters/`, `tools/`)
- Validation schemas and enforcement logic
- Lightweight index/summary docs pointing to evidence

## Sync Policy
1. Generate data with tools in this repo.
2. Copy artifacts to the data repo under matching paths.
3. Update `docs/state/evidence_index_2026-02-27.json` with canonical pointers.
