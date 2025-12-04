/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                                                                           ║
 * ║   Synth Task App · TypeScript                                             ║
 * ║   Banking77 Intent Classification                                         ║
 * ║                                                                           ║
 * ║   A reference implementation of the Synth Task App contract.              ║
 * ║   This app enables prompt optimization via the GEPA algorithm.            ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 *
 * Architecture
 * ────────────
 * This Task App exposes three endpoints that Synth uses during optimization:
 *
 *   GET  /health     → Liveness probe (unauthenticated)
 *   GET  /task_info  → Describes the task, dataset, and scoring rubric
 *   POST /rollout    → Executes one episode: render prompt → call LLM → score
 *
 * The optimization loop works as follows:
 *   1. Synth proposes a candidate prompt template
 *   2. Synth calls /rollout with that template + a seed
 *   3. This app renders the prompt, calls the LLM, scores the response
 *   4. Synth uses the score to evolve better prompts
 *
 * Running
 * ───────
 *   bun install && bun run dev     # Development (hot reload)
 *   bun run start                  # Production
 *
 * Environment
 * ───────────
 *   ENVIRONMENT_API_KEY   API key for authenticating Synth requests
 *   PORT                  Server port (default: 8001)
 */

import { Hono } from "hono";

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/** A labeled sample from the Banking77 dataset */
interface Sample {
  text: string;
  label: string;
}

/** Prompt section within a template (system, user, etc.) */
interface PromptSection {
  role: "system" | "user" | "assistant";
  content?: string;
  pattern?: string;
  order?: number;
}

/** Prompt template structure from Synth */
interface PromptTemplate {
  id?: string;
  sections?: PromptSection[];
  prompt_sections?: PromptSection[];
}

/** Rollout request from Synth */
interface RolloutRequest {
  run_id: string;
  env: {
    seed?: number;
    config?: Record<string, unknown>;
  };
  policy: {
    policy_id?: string;
    policy_name?: string;
    config: {
      model?: string;
      inference_url?: string;
      api_base?: string;
      base_url?: string;
      prompt_template?: PromptTemplate | string;
    };
  };
}

/** A single step in a trajectory */
interface Step {
  obs: Record<string, unknown>;
  tool_calls: ToolCall[];
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
}

/** Tool call from LLM response */
interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

/** Complete trajectory for one episode */
interface Trajectory {
  env_id: string;
  policy_id: string;
  steps: Step[];
  length: number;
  inference_url: string;
}

/** Rollout response to Synth */
interface RolloutResponse {
  run_id: string;
  trajectories: Trajectory[];
  metrics: {
    episode_returns: number[];
    mean_return: number;
    num_steps: number;
    num_episodes: number;
    outcome_score: number;
  };
  aborted: boolean;
  ops_executed: number;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Configuration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const config = {
  port: parseInt(process.env.PORT || "8001", 10),
  apiKey: process.env.ENVIRONMENT_API_KEY,
} as const;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Dataset
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const __dirname = dirname(fileURLToPath(import.meta.url));

interface Dataset {
  samples: Sample[];
  labels: string[];
}

function loadDataset(): Dataset {
  const path = join(__dirname, "../../data/banking77.json");
  try {
    const data = JSON.parse(readFileSync(path, "utf-8"));
    console.log(`📊 Loaded ${data.samples.length} samples`);
    return data;
  } catch {
    console.warn("⚠️  Dataset not found, using embedded samples");
    return {
      samples: [
        { text: "How do I reset my PIN?", label: "change_pin" },
        { text: "My card hasn't arrived yet", label: "card_arrival" },
        { text: "I want to cancel my card", label: "terminate_account" },
        { text: "How do I activate my new card?", label: "activate_my_card" },
        { text: "I need to dispute a transaction", label: "transaction_charged_twice" },
      ],
      labels: [
        "change_pin", "card_arrival", "terminate_account",
        "activate_my_card", "transaction_charged_twice",
      ],
    };
  }
}

const dataset = loadDataset();

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Prompt Rendering
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/** Render a template string with placeholder substitution: {key} → value */
function render(template: string, vars: Record<string, string>): string {
  return Object.entries(vars).reduce(
    (text, [key, value]) => text.replaceAll(`{${key}}`, value),
    template
  );
}

/** Build chat messages from a policy config and sample */
function buildMessages(
  policyConfig: RolloutRequest["policy"]["config"],
  sample: Sample
): Array<{ role: string; content: string }> {
  const vars = {
    query: sample.text,
    text: sample.text,
    intents: dataset.labels.join(", "),
  };

  const template = policyConfig.prompt_template;

  // String template (simple format)
  if (typeof template === "string") {
    return [
      { role: "system", content: render(template, vars) },
      { role: "user", content: `Query: ${sample.text}\nClassify using the classify tool.` },
    ];
  }

  // Object template with sections
  if (template) {
    const sections = template.prompt_sections || template.sections || [];
    if (sections.length > 0) {
      return [...sections]
        .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
        .map((s) => ({
          role: s.role,
          content: render(s.content || s.pattern || "", vars),
        }));
    }
  }

  // Default fallback
  return [
    { role: "system", content: "You are a banking assistant. Classify queries using the classify tool." },
    { role: "user", content: `Query: ${sample.text}\nIntents: ${vars.intents}\nClassify this query.` },
  ];
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LLM Client
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/** The classify tool that the LLM must call */
const CLASSIFY_TOOL = {
  type: "function" as const,
  function: {
    name: "classify",
    description: "Classify the customer query into a banking intent category",
    parameters: {
      type: "object",
      properties: {
        intent: { type: "string", description: "The classified intent" },
      },
      required: ["intent"],
    },
  },
};

/** Call an OpenAI-compatible LLM endpoint (Synth provides authenticated inference_url) */
async function callLLM(
  baseUrl: string,
  model: string,
  messages: Array<{ role: string; content: string }>
): Promise<{ prediction: string | null; toolCalls: ToolCall[] }> {
  // Construct the chat completions URL, preserving any query params
  const qIdx = baseUrl.indexOf("?");
  const url = qIdx === -1
    ? `${baseUrl.replace(/\/$/, "")}/chat/completions`
    : `${baseUrl.slice(0, qIdx).replace(/\/$/, "")}/chat/completions${baseUrl.slice(qIdx)}`;

  const headers: Record<string, string> = { "Content-Type": "application/json" };

  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      messages,
      tools: [CLASSIFY_TOOL],
      tool_choice: "required",
      temperature: 0,
      max_tokens: 100,
    }),
  });

  if (!res.ok) {
    throw new Error(`LLM error: ${res.status} ${await res.text()}`);
  }

  const data = await res.json();
  const choice = data.choices?.[0];
  const toolCalls: ToolCall[] = [];
  let prediction: string | null = null;

  // Extract tool calls and prediction
  for (const call of choice?.message?.tool_calls || []) {
    toolCalls.push({
      id: call.id,
      type: "function",
      function: { name: call.function.name, arguments: call.function.arguments },
    });
    if (call.function.name === "classify") {
      try {
        prediction = JSON.parse(call.function.arguments).intent;
      } catch { /* parse error, prediction stays null */ }
    }
  }

  // Fallback to raw content if no tool call
  prediction ??= choice?.message?.content?.trim() || null;

  return { prediction, toolCalls };
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HTTP Handlers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const app = new Hono();

/** Middleware: Verify API key for protected routes */
function requireAuth(c: any, next: () => Promise<void>) {
  if (config.apiKey && c.req.header("x-api-key") !== config.apiKey) {
    return c.json({ error: { code: "unauthorised", message: "API key missing or invalid" } }, 401);
  }
  return next();
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /health — Liveness probe
// ─────────────────────────────────────────────────────────────────────────────
app.get("/health", (c) => c.json({ healthy: true }));

// ─────────────────────────────────────────────────────────────────────────────
// GET /task_info — Task metadata and available seeds
// ─────────────────────────────────────────────────────────────────────────────
app.get("/task_info", requireAuth, (c) => {
  // Parse requested seeds from query string
  const url = new URL(c.req.url);
  const requested = [...url.searchParams.getAll("seed"), ...url.searchParams.getAll("seeds")]
    .map((s) => parseInt(s, 10))
    .filter((n) => !isNaN(n));

  const allSeeds = Array.from({ length: dataset.samples.length }, (_, i) => i);
  const seedGroups = requested.length > 0 ? requested.map((s) => [s]) : [allSeeds];

  return c.json(
    seedGroups.map((seeds) => ({
      task: {
        task_id: "banking77-typescript",
        name: "Banking77 Intent Classification",
        description: "Classify banking customer queries into intent categories",
        version: "1.0.0",
      },
      environment: "banking77",
      dataset: { seeds, train_count: dataset.samples.length, val_count: 0, test_count: 0 },
      rubric: {
        scoring_criteria: "exact_match",
        metric_primary: "accuracy",
        metric_range: [0.0, 1.0],
      },
      inference: { mode: "tool_call", supported_tools: ["classify"] },
      limits: { max_response_tokens: 100, timeout_seconds: 30 },
    }))
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// POST /rollout — Execute one classification episode
// ─────────────────────────────────────────────────────────────────────────────
app.post("/rollout", requireAuth, async (c) => {
  const req: RolloutRequest = await c.req.json();
  const seed = req.env.seed ?? 0;
  const sample = dataset.samples[seed % dataset.samples.length];

  // Resolve inference URL
  const inferenceUrl = req.policy.config.inference_url
    || req.policy.config.api_base
    || req.policy.config.base_url;

  if (!inferenceUrl) {
    return c.json({ error: "Missing inference_url in policy.config" }, 400);
  }

  const model = req.policy.config.model || "gpt-4o-mini";
  const messages = buildMessages(req.policy.config, sample);

  // Call LLM
  let prediction: string | null = null;
  let toolCalls: ToolCall[] = [];

  try {
    const result = await callLLM(inferenceUrl, model, messages);
    prediction = result.prediction;
    toolCalls = result.toolCalls;
  } catch (err) {
    console.error("❌ LLM call failed:", err);
    return c.json({ error: `LLM call failed: ${err}` }, 502);
  }

  // Score: exact match on intent label
  const correct = prediction?.toLowerCase() === sample.label.toLowerCase();
  const reward = correct ? 1.0 : 0.0;

  console.log(`🎯 seed=${seed} expected="${sample.label}" predicted="${prediction}" ${correct ? "✓" : "✗"}`);

  // Build response
  const response: RolloutResponse = {
    run_id: req.run_id,
    trajectories: [{
      env_id: `task::train::${seed}`,
      policy_id: req.policy.policy_id || req.policy.policy_name || "policy",
      inference_url: inferenceUrl,
      length: 1,
      steps: [{
        obs: { query: sample.text, index: seed },
        tool_calls: toolCalls,
        reward,
        done: true,
        info: { expected: sample.label, predicted: prediction, correct },
      }],
    }],
    metrics: {
      episode_returns: [reward],
      mean_return: reward,
      num_steps: 1,
      num_episodes: 1,
      outcome_score: reward,
    },
    aborted: false,
    ops_executed: 1,
  };

  return c.json(response);
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Server
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

console.log(`
╭───────────────────────────────────────╮
│  Synth Task App · Banking77           │
│  Port: ${String(config.port).padEnd(5)}                          │
│  Auth: ${config.apiKey ? "enabled ✓" : "disabled ⚠"}                        │
╰───────────────────────────────────────╯
`);

export default {
  port: config.port,
  fetch: app.fetch,
};
