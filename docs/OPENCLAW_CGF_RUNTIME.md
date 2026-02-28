# OpenClaw + CGF Runtime Integration

## Scope
This repo contains the OpenClaw governance plugin source:
- `plugins/cgf-governance/cgf-governance.ts`
- `plugins/cgf-governance/openclaw.plugin.json`

## Runtime Data Location
Governance evidence is written to the experimental-data repo (not source repo):
- `/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/openclaw_adapter_data/events.jsonl`

Local compatibility symlink (ignored by git):
- `/tmp/openclaws/Repos/host-adapters/openclaw_adapter_data -> /tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/openclaw_adapter_data`

## Active OpenClaw Config
Plugin is loaded from:
- `/tmp/openclaws/Repos/host-adapters/plugins/cgf-governance/cgf-governance.ts`

Plugin config is stored in local OpenClaw config:
- `/Users/meganpastore/.openclaw/openclaw.json`

## Gateway Autostart (Local Keepalive)
Because the stock `ai.openclaw.gateway` service was unstable in this environment, gateway autostart uses:
- LaunchAgent label: `ai.openclaw.gateway.keepalive`
- Plist: `/Users/meganpastore/Library/LaunchAgents/ai.openclaw.gateway.keepalive.plist`
- Script: `/Users/meganpastore/.openclaw/bin/openclaw_gateway_keepalive.sh`
- Logs:
  - `/Users/meganpastore/.openclaw/logs/gateway.keepalive.out.log`
  - `/Users/meganpastore/.openclaw/logs/gateway.keepalive.err.log`

## Local Runtime Patches Applied
The local OpenClaw installation has been patched for governance hook enforcement in HTTP tool invocation and session-store gating.
Backups created:
- `/opt/homebrew/lib/node_modules/openclaw/dist/sessions-Hkcy8tM7.js.bak_20260227_1928`
- `/opt/homebrew/lib/node_modules/openclaw/dist/gateway-cli-BSPSAjqx.js.bak_20260227_1929`
- `/opt/homebrew/lib/node_modules/openclaw/dist/gateway-cli-CD7BHA7a.js.bak_20260227_1940`

## Verification Commands
ALLOW path:
```bash
curl -sS -H "Authorization: Bearer $(jq -r '.gateway.auth.token' /Users/meganpastore/.openclaw/openclaw.json)" \
  -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:18789/tools/invoke \
  --data '{"tool":"memory_search","args":{"query":"health-check","limit":1}}'
```

Evidence check:
```bash
tail -n 10 /tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/openclaw_adapter_data/events.jsonl
```

Expected evidence includes:
- `adapter_registered`
- `decision_made`
- `outcome_logged`
