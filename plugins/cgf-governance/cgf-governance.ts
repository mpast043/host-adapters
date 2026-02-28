import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import type {
  OpenClawPluginApi,
  OpenClawPluginDefinition,
  PluginHookAfterToolCallEvent,
  PluginHookBeforeToolCallEvent,
  PluginHookToolContext,
} from "openclaw/plugin-sdk";

type JsonMap = Record<string, unknown>;

type FailModeRow = {
  action_type: string;
  risk_tier: string;
  fail_mode: "fail_open" | "fail_closed" | "defer";
  timeout_ms?: number;
};

type DecisionEnvelope = {
  proposalId: string;
  decisionId: string;
  actionType: "tool_call" | "memory_write";
  riskTier: "low" | "medium" | "high";
  startedAtMs: number;
};

type PluginConfig = {
  cgfEndpoint: string;
  authToken?: string;
  timeoutMs: number;
  schemaVersion: string;
  adapterType: string;
  dataDir: string;
  registerCooldownMs: number;
};

const DEFAULTS: PluginConfig = {
  cgfEndpoint: process.env.CGF_ENDPOINT?.trim() || "http://127.0.0.1:8082",
  authToken: process.env.CGF_AUTH_TOKEN?.trim() || undefined,
  timeoutMs: Number(process.env.CGF_TIMEOUT_MS || 700),
  schemaVersion: "0.3.0",
  adapterType: "openclaw",
  dataDir:
    process.env.CGF_DATA_DIR?.trim() ||
    "/tmp/openclaws/Repos/host-adapters-experimental-data/host-adapters/openclaw_adapter_data",
  registerCooldownMs: 30_000,
};

const TOOL_HIGH_RISK_RE =
  /(write|edit|delete|remove|exec|shell|bash|python|apply_patch|install|uninstall|restart|kill)/i;
const TOOL_LOW_RISK_RE = /(read|list|get|find|search|query|status|health|ls|cat|fetch)/i;

const state = {
  adapterId: "",
  registered: false,
  lastRegisterAttemptAtMs: 0,
  failModes: new Map<string, FailModeRow>(),
  pendingByKey: new Map<string, DecisionEnvelope[]>(),
};

function normalizeConfig(pluginConfig: Record<string, unknown> | undefined): PluginConfig {
  const cfg = pluginConfig ?? {};
  return {
    cgfEndpoint:
      (typeof cfg.cgfEndpoint === "string" && cfg.cgfEndpoint.trim()) || DEFAULTS.cgfEndpoint,
    authToken:
      (typeof cfg.authToken === "string" && cfg.authToken.trim()) || DEFAULTS.authToken,
    timeoutMs:
      typeof cfg.timeoutMs === "number" && Number.isFinite(cfg.timeoutMs) && cfg.timeoutMs > 0
        ? Math.floor(cfg.timeoutMs)
        : DEFAULTS.timeoutMs,
    schemaVersion:
      (typeof cfg.schemaVersion === "string" && cfg.schemaVersion.trim()) || DEFAULTS.schemaVersion,
    adapterType:
      (typeof cfg.adapterType === "string" && cfg.adapterType.trim()) || DEFAULTS.adapterType,
    dataDir: (typeof cfg.dataDir === "string" && cfg.dataDir.trim()) || DEFAULTS.dataDir,
    registerCooldownMs:
      typeof cfg.registerCooldownMs === "number" &&
      Number.isFinite(cfg.registerCooldownMs) &&
      cfg.registerCooldownMs >= 0
        ? Math.floor(cfg.registerCooldownMs)
        : DEFAULTS.registerCooldownMs,
  };
}

function ensureDir(dirPath: string): void {
  if (!existsSync(dirPath)) mkdirSync(dirPath, { recursive: true });
}

function stableStringify(value: unknown): string {
  const seen = new WeakSet<object>();

  const normalize = (input: unknown): unknown => {
    if (Array.isArray(input)) return input.map((entry) => normalize(entry));
    if (input && typeof input === "object") {
      if (seen.has(input as object)) return "[Circular]";
      seen.add(input as object);
      const out: Record<string, unknown> = {};
      for (const key of Object.keys(input as Record<string, unknown>).sort()) {
        out[key] = normalize((input as Record<string, unknown>)[key]);
      }
      return out;
    }
    return input;
  };

  return JSON.stringify(normalize(value));
}

function hashInput(value: unknown): string {
  return createHash("sha256").update(stableStringify(value), "utf8").digest("hex");
}

function nowSec(): number {
  return Date.now() / 1000;
}

function keyForPending(sessionKey: string | undefined, toolName: string, params: unknown): string {
  return `${sessionKey || "session:unknown"}::${toolName}::${hashInput(params).slice(0, 20)}`;
}

function logEvidence(config: PluginConfig, eventType: string, payload: JsonMap): void {
  try {
    ensureDir(config.dataDir);
    const event = {
      schema_version: config.schemaVersion,
      event_type: eventType,
      adapter_id: state.adapterId || "unregistered",
      timestamp: nowSec(),
      payload,
    };
    appendFileSync(path.join(config.dataDir, "events.jsonl"), `${JSON.stringify(event)}\n`, "utf8");
  } catch {
    // Never crash runtime on evidence write failures.
  }
}

async function fetchJson(
  config: PluginConfig,
  endpointPath: string,
  body: JsonMap,
): Promise<JsonMap> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (config.authToken) headers.Authorization = `Bearer ${config.authToken}`;
    const response = await fetch(`${config.cgfEndpoint}${endpointPath}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${endpointPath} failed (${response.status}): ${text || response.statusText}`);
    }
    return (await response.json()) as JsonMap;
  } finally {
    clearTimeout(timeout);
  }
}

function classifyToolRisk(toolName: string): "low" | "medium" | "high" {
  if (TOOL_HIGH_RISK_RE.test(toolName)) return "high";
  if (TOOL_LOW_RISK_RE.test(toolName)) return "low";
  return "medium";
}

function sideEffectsHint(toolName: string): string[] {
  if (TOOL_HIGH_RISK_RE.test(toolName)) return ["write"];
  if (TOOL_LOW_RISK_RE.test(toolName)) return ["read"];
  return ["network"];
}

function resolveFailMode(
  actionType: "tool_call" | "memory_write",
  riskTier: "low" | "medium" | "high",
): FailModeRow {
  const key = `${actionType}:${riskTier}`;
  const fromTable = state.failModes.get(key);
  if (fromTable) return fromTable;
  if (actionType === "memory_write" && riskTier !== "low") {
    return { action_type: actionType, risk_tier: riskTier, fail_mode: "fail_closed", timeout_ms: 500 };
  }
  if (riskTier === "high") {
    return { action_type: actionType, risk_tier: riskTier, fail_mode: "fail_closed", timeout_ms: 500 };
  }
  if (riskTier === "medium") {
    return { action_type: actionType, risk_tier: riskTier, fail_mode: "defer", timeout_ms: 500 };
  }
  return { action_type: actionType, risk_tier: riskTier, fail_mode: "fail_open", timeout_ms: 500 };
}

function applyFailMode(
  actionType: "tool_call" | "memory_write",
  riskTier: "low" | "medium" | "high",
): { allow: boolean; reason: string } {
  const failMode = resolveFailMode(actionType, riskTier);
  if (failMode.fail_mode === "fail_open") {
    return { allow: true, reason: `CGF unavailable; ${actionType} allowed by fail_open` };
  }
  if (failMode.fail_mode === "defer") {
    return { allow: false, reason: `CGF unavailable; ${actionType} deferred` };
  }
  return { allow: false, reason: `CGF unavailable; ${actionType} blocked by fail_closed` };
}

async function ensureRegistered(config: PluginConfig, api: OpenClawPluginApi): Promise<void> {
  const now = Date.now();
  if (state.registered) return;
  if (now - state.lastRegisterAttemptAtMs < config.registerCooldownMs) return;
  state.lastRegisterAttemptAtMs = now;

  try {
    const response = await fetchJson(config, "/v1/register", {
      schema_version: config.schemaVersion,
      adapter_type: config.adapterType,
      host_config: {
        host_type: "openclaw",
        namespace: "default",
        capabilities: ["tool_call", "memory_write"],
        version: "0.1.0",
      },
      features: ["before_tool_call", "after_tool_call", "session_store_write"],
      risk_tiers: {
        high: "fail_closed",
        medium: "defer",
        low: "fail_open",
      },
      timestamp: nowSec(),
    });

    const adapterId = typeof response.adapter_id === "string" ? response.adapter_id : "";
    if (!adapterId) throw new Error("CGF registration missing adapter_id");

    state.adapterId = adapterId;
    state.registered = true;
    state.failModes.clear();

    const table = Array.isArray(response.fail_mode_table)
      ? (response.fail_mode_table as FailModeRow[])
      : [];
    for (const row of table) {
      if (!row?.action_type || !row?.risk_tier || !row?.fail_mode) continue;
      state.failModes.set(`${row.action_type}:${row.risk_tier}`, row);
    }

    api.logger.info(`[cgf-governance] registered adapter_id=${state.adapterId}`);
    logEvidence(config, "adapter_registered", {
      adapter_id: state.adapterId,
      fail_mode_rows: state.failModes.size,
    });
  } catch (error) {
    state.registered = false;
    api.logger.warn(`[cgf-governance] registration failed: ${String(error)}`);
    logEvidence(config, "cgf_unreachable", {
      stage: "register",
      error: String(error),
    });
  }
}

async function evaluateToolCall(
  config: PluginConfig,
  api: OpenClawPluginApi,
  event: PluginHookBeforeToolCallEvent,
  ctx: PluginHookToolContext,
): Promise<{ decision: string; decisionId: string; proposalId: string; maybeParams?: JsonMap }> {
  const riskTier = classifyToolRisk(event.toolName);
  const proposalId = `prop-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const paramsHash = hashInput(event.params);

  const requestBody: JsonMap = {
    schema_version: config.schemaVersion,
    adapter_id: state.adapterId || null,
    host_config: {
      host_type: "openclaw",
      namespace: "default",
      capabilities: ["tool_call", "memory_write"],
      version: "0.1.0",
    },
    proposal: {
      schema_version: config.schemaVersion,
      proposal_id: proposalId,
      timestamp: nowSec(),
      action_type: "tool_call",
      action_params: {
        tool_name: event.toolName,
        tool_args_hash: paramsHash,
        estimated_tokens: Math.ceil(stableStringify(event.params).length / 4),
        side_effects_hint: sideEffectsHint(event.toolName),
      },
      context_refs: [ctx.sessionKey || "session:unknown", ctx.agentId || "agent:unknown"],
      estimated_cost: {
        tokens: Math.ceil(stableStringify(event.params).length / 4),
      },
      risk_tier: riskTier,
      priority: 0,
    },
    context: {
      schema_version: config.schemaVersion,
      agent_id: ctx.agentId || null,
      session_id: ctx.sessionKey || null,
      turn_number: 0,
      recent_errors: 0,
      memory_growth_rate: 0,
    },
    capacity_signals: {
      schema_version: config.schemaVersion,
      token_rate: 0,
      tool_call_rate: 0,
      error_rate: 0,
      memory_growth: 0,
    },
  };

  try {
    const response = await fetchJson(config, "/v1/evaluate", requestBody);
    const decisionObj = (response.decision || {}) as JsonMap;
    const decision = typeof decisionObj.decision === "string" ? decisionObj.decision : "DEFER";
    const decisionId =
      (typeof decisionObj.decision_id === "string" && decisionObj.decision_id) ||
      `dec-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const maybeConstraint = decisionObj.constraint as JsonMap | undefined;

    logEvidence(config, "decision_made", {
      proposal_id: proposalId,
      decision_id: decisionId,
      decision,
      tool_name: event.toolName,
      session_key: ctx.sessionKey || null,
    });

    let maybeParams: JsonMap | undefined;
    if (decision === "CONSTRAIN" && maybeConstraint?.type === "drop_params_keys") {
      const keys = Array.isArray(maybeConstraint.params)
        ? []
        : (maybeConstraint.params as JsonMap | undefined)?.keys;
      if (Array.isArray(keys)) {
        const next = { ...(event.params || {}) };
        for (const key of keys) {
          if (typeof key === "string") delete next[key];
        }
        maybeParams = next;
      }
    }

    return { decision, decisionId, proposalId, maybeParams };
  } catch (error) {
    const fail = applyFailMode("tool_call", riskTier);
    api.logger.warn(`[cgf-governance] evaluate failed for tool=${event.toolName}: ${String(error)}`);
    logEvidence(config, "cgf_unreachable", {
      stage: "evaluate_tool_call",
      tool_name: event.toolName,
      risk_tier: riskTier,
      fail_mode_allow: fail.allow,
      error: String(error),
    });
    if (fail.allow) {
      return {
        decision: "ALLOW",
        decisionId: `dec-fail-open-${Date.now()}`,
        proposalId,
      };
    }
    return {
      decision: "BLOCK",
      decisionId: `dec-fail-${Date.now()}`,
      proposalId,
    };
  }
}

function enqueuePending(key: string, envelope: DecisionEnvelope): void {
  const queue = state.pendingByKey.get(key) ?? [];
  queue.push(envelope);
  state.pendingByKey.set(key, queue);
}

function dequeuePending(key: string): DecisionEnvelope | undefined {
  const queue = state.pendingByKey.get(key);
  if (!queue || queue.length === 0) return undefined;
  const next = queue.shift();
  if (queue.length === 0) state.pendingByKey.delete(key);
  else state.pendingByKey.set(key, queue);
  return next;
}

async function reportOutcome(
  config: PluginConfig,
  api: OpenClawPluginApi,
  envelope: DecisionEnvelope,
  success: boolean,
  summary: string,
  errors: string[],
): Promise<void> {
  if (!state.adapterId) return;
  try {
    await fetchJson(config, "/v1/outcomes/report", {
      schema_version: config.schemaVersion,
      adapter_id: state.adapterId,
      proposal_id: envelope.proposalId,
      decision_id: envelope.decisionId,
      executed: true,
      executed_at: nowSec(),
      duration_ms: Date.now() - envelope.startedAtMs,
      success,
      committed: success,
      quarantined: false,
      errors,
      result_summary: summary,
    });
    logEvidence(config, "outcome_logged", {
      proposal_id: envelope.proposalId,
      decision_id: envelope.decisionId,
      success,
      summary,
    });
  } catch (error) {
    api.logger.warn(`[cgf-governance] outcome report failed: ${String(error)}`);
    logEvidence(config, "errors", {
      stage: "report_outcome",
      error: String(error),
      proposal_id: envelope.proposalId,
      decision_id: envelope.decisionId,
    });
  }
}

function estimateStoreSensitivity(storeJsonBytes: number): "low" | "medium" | "high" {
  if (storeJsonBytes > 10_000_000) return "high";
  if (storeJsonBytes > 1_000_000) return "medium";
  return "low";
}

function installMemoryWriteGate(config: PluginConfig, api: OpenClawPluginApi): void {
  const globalObj = globalThis as Record<string, unknown>;

  globalObj.__openclawCgfMemoryGate = async (params: {
    storePath: string;
    store: Record<string, unknown>;
    opts?: Record<string, unknown>;
  }) => {
    await ensureRegistered(config, api);

    const storeString = stableStringify(params.store || {});
    const sizeBytes = Buffer.byteLength(storeString, "utf8");
    const riskTier = estimateStoreSensitivity(sizeBytes);
    const proposalId = `prop-mem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const requestBody: JsonMap = {
      schema_version: config.schemaVersion,
      adapter_id: state.adapterId || null,
      host_config: {
        host_type: "openclaw",
        namespace: "default",
        capabilities: ["memory_write"],
        version: "0.1.0",
      },
      proposal: {
        schema_version: config.schemaVersion,
        proposal_id: proposalId,
        timestamp: nowSec(),
        action_type: "memory_write",
        action_params: {
          namespace: path.dirname(params.storePath || "/unknown"),
          size_bytes: sizeBytes,
          sensitivity_hint: riskTier,
          content_hash: hashInput(params.store).slice(0, 64),
          operation: "update",
        },
        context_refs: [
          (params.opts?.activeSessionKey as string) || "session:unknown",
          state.adapterId || "adapter:unknown",
        ],
        estimated_cost: {
          bytes: sizeBytes,
        },
        risk_tier: riskTier,
        priority: 0,
      },
      context: {
        schema_version: config.schemaVersion,
        agent_id: null,
        session_id: (params.opts?.activeSessionKey as string) || null,
        turn_number: 0,
        recent_errors: 0,
        memory_growth_rate: sizeBytes,
      },
      capacity_signals: {
        schema_version: config.schemaVersion,
        token_rate: 0,
        tool_call_rate: 0,
        error_rate: 0,
        memory_growth: sizeBytes,
      },
    };

    try {
      const response = await fetchJson(config, "/v1/evaluate", requestBody);
      const decisionObj = (response.decision || {}) as JsonMap;
      const decision = typeof decisionObj.decision === "string" ? decisionObj.decision : "DEFER";
      const decisionId =
        (typeof decisionObj.decision_id === "string" && decisionObj.decision_id) ||
        `dec-mem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const envelope: DecisionEnvelope = {
        proposalId,
        decisionId,
        actionType: "memory_write",
        riskTier,
        startedAtMs: Date.now(),
      };

      if (decision === "ALLOW" || decision === "AUDIT" || decision === "CONSTRAIN") {
        return { allowed: true, envelope };
      }
      return {
        allowed: false,
        reason: `CGF ${decision} memory_write`,
        envelope,
      };
    } catch (error) {
      const fail = applyFailMode("memory_write", riskTier);
      logEvidence(config, "cgf_unreachable", {
        stage: "evaluate_memory_write",
        risk_tier: riskTier,
        fail_mode_allow: fail.allow,
        error: String(error),
      });
      return {
        allowed: fail.allow,
        reason: fail.reason,
      };
    }
  };

  globalObj.__openclawCgfMemoryReportOutcome = async (params: {
    gate?: { envelope?: DecisionEnvelope };
    success: boolean;
    error?: string;
  }) => {
    const envelope = params.gate?.envelope;
    if (!envelope) return;
    await reportOutcome(
      config,
      api,
      envelope,
      params.success,
      params.success ? "session store write committed" : "session store write failed",
      params.error ? [params.error] : [],
    );
  };
}

function uninstallMemoryWriteGate(): void {
  const globalObj = globalThis as Record<string, unknown>;
  delete globalObj.__openclawCgfMemoryGate;
  delete globalObj.__openclawCgfMemoryReportOutcome;
}

function register(api: OpenClawPluginApi): void {
  const config = normalizeConfig(api.pluginConfig as Record<string, unknown> | undefined);
  ensureDir(config.dataDir);

  api.on("gateway_start", async () => {
    await ensureRegistered(config, api);
    installMemoryWriteGate(config, api);
    api.logger.info(
      `[cgf-governance] gateway_start cgfEndpoint=${config.cgfEndpoint} dataDir=${config.dataDir}`,
    );
  });

  api.on("gateway_stop", () => {
    uninstallMemoryWriteGate();
  });

  api.on("before_tool_call", async (event, ctx) => {
    await ensureRegistered(config, api);

    const evaluated = await evaluateToolCall(config, api, event, ctx);
    const pendingKey = keyForPending(ctx.sessionKey, event.toolName, event.params);
    enqueuePending(pendingKey, {
      proposalId: evaluated.proposalId,
      decisionId: evaluated.decisionId,
      actionType: "tool_call",
      riskTier: classifyToolRisk(event.toolName),
      startedAtMs: Date.now(),
    });

    if (evaluated.decision === "BLOCK" || evaluated.decision === "DEFER") {
      return {
        block: true,
        blockReason: `CGF ${evaluated.decision}: tool ${event.toolName}`,
      };
    }

    if (evaluated.decision === "CONSTRAIN" && evaluated.maybeParams) {
      return {
        params: evaluated.maybeParams,
      };
    }

    return;
  });

  api.on("after_tool_call", async (event: PluginHookAfterToolCallEvent, ctx: PluginHookToolContext) => {
    const pendingKey = keyForPending(ctx.sessionKey, event.toolName, event.params);
    const envelope = dequeuePending(pendingKey);
    if (!envelope) return;
    await reportOutcome(
      config,
      api,
      envelope,
      !event.error,
      event.error ? "tool execution failed" : "tool execution succeeded",
      event.error ? [event.error] : [],
    );
  });
}

const plugin: OpenClawPluginDefinition = {
  id: "cgf-governance",
  name: "CGF Governance Runtime",
  description:
    "Runtime governance for OpenClaw tool calls and session-store writes via CGF.",
  version: "0.1.0",
  register,
};

export default plugin;
