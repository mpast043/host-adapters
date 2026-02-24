# Host Adapters Repository

CGF Host Adapters for OpenClaw, LangGraph, and other host systems.

## Structure

```
.
├── sdk/                      # CGF SDK (v0.4.0)
│   ├── cgf_client.py
│   ├── adapter_base.py
│   ├── errors.py
│   └── spec/
│       └── event_types.json
├── adapters/                 # Host-specific adapters
│   ├── openclaw_adapter_v01.py
│   ├── openclaw_adapter_v02.py
│   ├── langgraph_adapter_v01.py
│   └── openclaw_cgf_hook_v02.mjs
├── server/                   # CGF Server
│   ├── cgf_schemas_v03.py
│   └── cgf_server_v03.py
├── tools/                    # Testing & validation tools
│   ├── schema_lint.py
│   ├── contract_compliance_tests.py
│   ├── replay_governance_timeline.py
│   └── run_contract_suite.sh
│   └── validate_sdk_artifacts.py
├── policy/                   # Policy configuration
│   └── policy_config_v03.json
├── DEV.md                    # Developer guide
├── requirements.txt          # Python dependencies
└── Makefile                  # Build automation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test

# Run lint
make lint
```

## Schema Version

This repository uses **Schema v0.3.0** for all event and payload types.

## Environment Variables

- `CGF_ENDPOINT` - CGF server URL (default: http://127.0.0.1:8080)
- `CGF_TIMEOUT_MS` - Request timeout in ms (default: 500)
- `CGF_DATA_DIR` - Local data directory

## License

MIT