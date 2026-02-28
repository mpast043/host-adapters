# Local Compute MCP Target

This repository includes a local stdio MCP server for constrained compute tasks:

- Server script: `tools/local_compute_mcp.py`
- Default data/log root: `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/openclaw_adapter_data/compute_jobs`

## Tools

- `compute_health`
- `compute_exec`
- `compute_exec_background`
- `compute_job_status`
- `compute_job_tail`
- `compute_list_jobs`
- `compute_cancel_job`

## Safety Controls

- Allowed binaries: `python`, `python3`, `pytest`, `make`
- Allowed make targets: `test`, `test-fast`, `lint`, `format`, `workflow-auto`, `workflow-audit`, `openclaw-opt-check`
- Working directory is constrained to:
  - `/tmp/openclaws/Repos/host-adapters`
  - `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters`

## Registration

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.local-compute]
command = "/tmp/openclaws/Repos/host-adapters/.venv/bin/python"
args = ["/tmp/openclaws/Repos/host-adapters/tools/local_compute_mcp.py"]

[mcp_servers.local-compute.env]
LOCAL_COMPUTE_REPO_ROOT = "/tmp/openclaws/Repos/host-adapters"
HOST_ADAPTERS_EXPERIMENTAL_DATA_DIR = "/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters"
```

## Quick Check

```bash
mcporter list
mcporter call local-compute.compute_health
mcporter call local-compute.compute_exec --args '{"argv":["python3","--version"]}'
```
