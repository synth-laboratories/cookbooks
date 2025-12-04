# Polyglot Task Apps

Banking77 intent-classification Task App implementations in multiple languages (Rust, Go, TypeScript) for Synth prompt optimization with MIPRO/GEPA.

## What you'll build
- An HTTP Task App that serves `/health`, `/task_info`, and `/rollout`
- Uses the shared `data/banking77.json` dataset (100 samples) to score prompts
- Deployable locally or via tunnel; compatible with Synth prompt-optimization jobs

## Prerequisites
- Language toolchains: Cargo (Rust), Go 1.21+, Bun 1.0+ (TypeScript)
- API keys: `ENVIRONMENT_API_KEY` (task app auth)
- Optional: `cloudflared` (or any HTTPS tunnel) to expose localhost

## Quick start (pick one)
- Rust: `cd rust && cargo run --release`
- Go: `cd go && go run .`
- TypeScript: `cd typescript && bun install && bun run dev`

Expose via tunnel (example):
```bash
cloudflared tunnel --url http://localhost:8001
# use the HTTPS URL as task_app_url for MIPRO/GEPA jobs
```

Start an optimization job (MIPRO example):
```bash
curl -X POST https://agent-learning.onrender.com/api/prompt-learning/online/jobs \
  -H "Authorization: Bearer $SYNTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "mipro",
    "config_body": {
      "prompt_learning": {
        "task_app_url": "https://your-tunnel-url",
        "task_app_api_key": "your-env-key"
      }
    }
  }'
```

## What is a Task App?

A Task App is an HTTP service that evaluates prompts for Synth's MIPRO and GEPA optimizers. The optimizer calls your `/rollout` endpoint with candidate prompts, and you return a reward indicating how well each prompt performed.

```
┌─────────────────┐         ┌──────────────────┐
│  MIPRO/GEPA     │  HTTP   │  Your Task App   │
│  Optimizer      │ ──────> │  (any language)  │
│                 │         │                  │
│  Proposes new   │         │  Evaluates the   │
│  prompts        │ <────── │  prompt, returns │
│                 │  reward │  reward          │
└─────────────────┘         └──────────────────┘
```

## Available Examples

| Language | Framework | Dependencies | Notes |
|----------|-----------|--------------|-------|
| [Rust](./rust/) | Axum | axum, tokio, reqwest | Fast, type-safe |
| [Go](./go/) | stdlib | None | Single binary, no deps |
| [TypeScript](./typescript/) | Hono + Bun | hono | Tested with GEPA (80% accuracy) |

## Quick Start

### 1. Choose a Language

```bash
cd rust/      # or go/, typescript/
```

### 2. Build and Run

**Rust:**
```bash
cargo run --release
```

**Go:**
```bash
go build && ./synth-task-app
```

**TypeScript:**
```bash
bun install && bun run dev
```

### 3. Expose via Tunnel

```bash
cloudflared tunnel --url http://localhost:8001
# Note the URL: https://random-words.trycloudflare.com
```

### 4. Start Optimization (No Python Required!)

```bash
curl -X POST https://agent-learning.onrender.com/api/prompt-learning/online/jobs \
  -H "Authorization: Bearer $SYNTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "mipro",
    "config_body": {
      "prompt_learning": {
        "task_app_url": "https://random-words.trycloudflare.com",
        "task_app_api_key": "your-env-key"
      }
    }
  }'
```

## Contract

All examples implement the same OpenAPI contract:

**Required Endpoints:**
- `GET /health` - Health check (unauthenticated OK)
- `GET /task_info` - Dataset metadata (authenticated)
- `POST /rollout` - Evaluate a prompt (authenticated)

**Key Request Fields:**
- `env.seed` - Dataset index
- `policy.config.inference_url` - LLM endpoint (see notes below)
- `policy.config.prompt_template` - The prompt to evaluate

**Key Response Fields:**
- `metrics.mean_return` - Reward (0.0-1.0) that drives optimization
- `trajectories[].steps[].reward` - Per-step reward

See [`synth_ai/contracts/task_app.yaml`](../../synth_ai/contracts/task_app.yaml) for the full OpenAPI specification.

## Authentication

Task Apps involve **two separate authentication flows**:

### 1. Task App Authentication (`X-API-Key`)

Requests *to* your task app from the optimizer include an `X-API-Key` header. This authenticates the optimizer to your task app.

```bash
# Set this when starting your task app
export ENVIRONMENT_API_KEY=your-secret-key
```

Your task app should verify `X-API-Key` matches `ENVIRONMENT_API_KEY` on `/rollout` and `/task_info` endpoints.

### 2. LLM API Authentication (`Authorization: Bearer`)

When your task app makes requests *to* OpenAI/Groq/etc, you need to add a Bearer token:

```bash
# Set this when starting your task app
export OPENAI_API_KEY=sk-...    # or
export GROQ_API_KEY=gsk_...
```

Your task app should read this from the environment and add it to LLM requests:
```
Authorization: Bearer sk-...
```

**Important:** The `X-API-Key` header from the optimizer is for task app auth only - do NOT forward it to the LLM API.

## Handling `inference_url`

The `policy.config.inference_url` field specifies the LLM endpoint base URL. It may contain query parameters:

```
https://api.openai.com/v1                    # Simple case
https://api.openai.com/v1?model=gpt-4o-mini  # With query params
```

When constructing the full endpoint URL, preserve any query string:

```
# WRONG - query params end up in wrong place
https://api.openai.com/v1?model=gpt-4o-mini/chat/completions

# CORRECT - insert path before query string
https://api.openai.com/v1/chat/completions?model=gpt-4o-mini
```

Example URL construction:
```python
if "?" in inference_url:
    base, query = inference_url.split("?", 1)
    url = f"{base.rstrip('/')}/chat/completions?{query}"
else:
    url = f"{inference_url.rstrip('/')}/chat/completions"
```

## Shared Dataset

All examples load from the same dataset file:

```
data/banking77.json
```

This contains 100 Banking77-style samples for intent classification.

## Project Structure

```
polyglot/
├── README.md           # This file
├── data/
│   └── banking77.json  # Shared dataset
├── rust/
│   ├── Cargo.toml
│   ├── src/main.rs
│   └── README.md
├── go/
│   ├── go.mod
│   ├── main.go
│   └── README.md
└── typescript/
    ├── package.json
    ├── src/index.ts
    └── README.md
```

## Customizing for Your Task

1. **Replace the dataset** - Edit `data/banking77.json` or load your own
2. **Update the tool schema** - Modify the `classify` tool to match your output format
3. **Adjust reward computation** - Change how you compare predictions to ground truth

## See Also

- [API Documentation](../../synth_ai/contracts/task_app.yaml) - Full OpenAPI spec
- [Python Task Apps](../task_apps/) - Python implementations
- [Synth API Docs](https://docs.usesynth.ai) - Full platform documentation
