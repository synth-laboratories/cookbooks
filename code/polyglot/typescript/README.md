# TypeScript Task App Example

A minimal but complete Task App implementation in TypeScript for Synth prompt optimization.
**Tested end-to-end with GEPA optimizer (80% accuracy on Banking77).**

## Features

- Uses [Hono](https://hono.dev/) - fast, lightweight web framework
- Loads dataset from shared JSON file (with embedded fallback)
- Implements `/health`, `/task_info`, and `/rollout` endpoints per OpenAPI contract
- Prompt template rendering with `{placeholder}` substitution
- Proper URL construction with query parameter handling
- Uses Bun runtime (fast, native TypeScript support)

## Quick Start

```bash
# Install dependencies
bun install

# Run in development (with hot reload)
bun run dev

# Run production
bun run start

# With authentication
ENVIRONMENT_API_KEY=your-secret bun run dev

# Custom port
PORT=3000 bun run dev
```

## Testing

```bash
# Health check
curl http://localhost:8001/health

# Manual rollout
curl -X POST http://localhost:8001/rollout \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "run_id": "test-1",
    "env": {"seed": 0},
    "policy": {
      "config": {
        "model": "gpt-4o-mini",
        "inference_url": "https://api.openai.com/v1"
      }
    },
    "mode": "eval"
  }'
```

## Running with Synth Optimizer

### Local Development (Recommended for Testing)

1. **Start the local backend** (from monorepo):
   ```bash
   cd monorepo && bash scripts/run_backend_local.sh
   # Starts: Redis, sqld, uvicorn on port 8000
   ```

2. **Start the task app:**
   ```bash
   bun install
   ENVIRONMENT_API_KEY=test-polyglot-key bun run dev
   ```

3. **Submit a job** using the example config:
   ```bash
   curl -X POST "http://localhost:8000/api/prompt-learning/online/jobs" \
     -H "Authorization: Bearer $SYNTH_API_KEY" \
     -H "Content-Type: application/json" \
     -d @../mipro_job.json
   ```

### Production (via Cloudflare Tunnel)

1. **Install and run:**
   ```bash
   bun install
   ENVIRONMENT_API_KEY=my-secret bun run dev
   ```

2. **Expose via Cloudflare tunnel:**
   ```bash
   cloudflared tunnel --url http://localhost:8001
   ```

3. **Start optimization:**
   ```bash
   curl -X POST https://agent-learning.onrender.com/api/prompt-learning/online/jobs \
     -H "Authorization: Bearer $SYNTH_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "algorithm": "mipro",
       "config_body": {
         "prompt_learning": {
           "task_app_url": "https://your-tunnel.trycloudflare.com",
           "task_app_api_key": "my-secret"
         }
       }
     }'
   ```

## Critical Implementation Details

### URL Construction with Query Parameters

The `inference_url` provided by the optimizer includes query parameters for tracing:
```
http://localhost:8000/api/interceptor/v1/trial-id?cid=trace_xxx
```

When appending `/chat/completions`, the path must come BEFORE the query string:

```typescript
// CORRECT: path before query
let url: string;
const queryIndex = inferenceUrl.indexOf("?");
if (queryIndex !== -1) {
  const base = inferenceUrl.slice(0, queryIndex).replace(/\/$/, "");
  const query = inferenceUrl.slice(queryIndex);
  url = `${base}/chat/completions${query}`;
} else {
  url = `${inferenceUrl.replace(/\/$/, "")}/chat/completions`;
}

// Result: http://host/path/chat/completions?cid=xxx
```

**Wrong approach** (causes 404):
```typescript
// WRONG: appends path after query string
const url = `${inferenceUrl}/chat/completions`;
// Result: http://host/path?cid=xxx/chat/completions  <-- 404!
```

### Handling Multiple Seeds in /task_info

The backend sends seed parameters as repeated keys: `?seeds=0&seeds=1&seeds=2`.
Parse both `seed` and `seeds` variants using URLSearchParams:

```typescript
const url = new URL(c.req.url);
const seedParams = url.searchParams.getAll("seed");
const seedsParams = url.searchParams.getAll("seeds");
const requestedSeeds = [...seedParams, ...seedsParams]
  .map((s) => parseInt(s, 10))
  .filter((n) => !isNaN(n));
```

## Deploying to Cloudflare Workers

The Hono framework supports Cloudflare Workers out of the box:

1. **Install Wrangler:**
   ```bash
   bun install -g wrangler
   ```

2. **Create `wrangler.toml`:**
   ```toml
   name = "synth-task-app"
   main = "src/index.ts"
   compatibility_date = "2024-01-01"

   [vars]
   ENVIRONMENT_API_KEY = "your-secret"
   ```

3. **Modify for Workers (change server start):**
   ```typescript
   // Replace the Bun serve export with:
   export default app;
   ```

4. **Deploy:**
   ```bash
   wrangler deploy
   ```

## Project Structure

```
typescript/
├── package.json
├── tsconfig.json
├── src/
│   └── index.ts    # Task app implementation
└── README.md
```

## Contract Reference

See [`synth_ai/contracts/task_app.yaml`](../../../synth_ai/contracts/task_app.yaml) for the full OpenAPI specification.