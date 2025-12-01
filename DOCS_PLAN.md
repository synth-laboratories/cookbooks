# Mintlify Cookbook Docs Plan

OpenAI-style cookbooks: code-first, real links to source files, minimal prose.

---

## Format Guidelines

**OpenAI cookbook style:**
- Start with runnable code block
- Link to source files inline: `[run_walkthrough.py](link)`
- Show real output/results
- Minimal explanation between code blocks
- Prerequisites as bullet list, not paragraphs
- No fluff headers like "Introduction" or "Conclusion"

**Link pattern:**
```
[filename.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/path/filename.py)
```

---

## 1. In-Process GEPA/MIPRO

**File:** `monorepo/docs/cookbooks/prompt-optimization-in-process.mdx`

### Draft Structure

```mdx
---
title: "In-Process Prompt Optimization"
description: "Run GEPA or MIPRO from a single Python script"
---

## Prerequisites

- Python 3.11+
- `uv` package manager
- API keys: `SYNTH_API_KEY`, `GROQ_API_KEY`

## GEPA: Banking77

Run optimization with one command:

\`\`\`bash
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/gepa/run_walkthrough.py
\`\`\`

**Source:** [run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/run_walkthrough.py) | [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/config.toml)

### What happens

1. Task app starts in background thread
2. Cloudflare tunnel created automatically  
3. GEPA job submitted and polled
4. Results returned, cleanup automatic

### Results

Job `pl_5ea04259c2fd4c7a` — **83.33% accuracy** in ~35s

\`\`\`json
{
  "job_id": "pl_5ea04259c2fd4c7a",
  "best_score": 0.8333,
  "algorithm": "gepa",
  "total_time_seconds": 35.6
}
\`\`\`

**Full results:** [results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/results.json)

---

## MIPRO: Banking77

\`\`\`bash
uv run python /path/to/cookbooks/code/training/prompt_learning/mipro/run_walkthrough.py
\`\`\`

**Source:** [run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/run_walkthrough.py) | [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/config.toml)

### Results

Job `pl_e95cc778c0fb4742` — **60% accuracy** in ~130s

**Full results:** [results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/results.json)

---

## Core Pattern

\`\`\`python
from synth_ai.task import InProcessTaskApp
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

async with InProcessTaskApp(
    task_app_path="task_app.py",
    port=8001,
) as task_app:
    job = PromptLearningJob.from_config(
        config_path="config.toml",
        task_app_url=task_app.url,
    )
    results = await job.poll_until_complete()
\`\`\`

**Full walkthrough:** [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/walkthrough.md)
```

---

## 2. Polyglot Task Apps

**File:** `monorepo/docs/cookbooks/prompt-optimization-polyglot.mdx`

### Draft Structure

```mdx
---
title: "Polyglot Task Apps"
description: "Prompt optimization with task apps in Rust, Go, TypeScript, Python"
---

Task apps work in any language. Build, run, optimize via CLI.

## Results Summary

| Language   | Job ID                 | Accuracy | Source |
|------------|------------------------|----------|--------|
| Rust       | `pl_4f69a1b099a14e4b`  | **100%** | [main.rs](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/src/main.rs) |
| TypeScript | `pl_787c47998cfe4745`  | 85.7%    | [index.ts](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/src/index.ts) |
| Python     | `pl_7e0227cc41454ec5`  | 66.7%    | [app.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/app.py) |
| Go         | `pl_1dd94dfdc8c6479d`  | 60%      | [main.go](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/main.go) |

---

## Rust

\`\`\`bash
cd cookbooks/code/polyglot/rust
cargo build --release
ENVIRONMENT_API_KEY=secret ./target/release/synth-task-app
\`\`\`

Expose via tunnel:

\`\`\`bash
cloudflared tunnel --url http://localhost:8001
\`\`\`

Run optimization:

\`\`\`bash
synth train gepa_config.toml --poll
\`\`\`

**Source:** [main.rs](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/src/main.rs) | [gepa_config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/gepa_config.toml) | [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/walkthrough.md)

---

## TypeScript

\`\`\`bash
cd cookbooks/code/polyglot/typescript
npm install && npm run dev
\`\`\`

**Source:** [index.ts](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/src/index.ts) | [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/walkthrough.md)

---

## Go

\`\`\`bash
cd cookbooks/code/polyglot/go
go build -o synth-task-app && ./synth-task-app
\`\`\`

**Source:** [main.go](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/main.go) | [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/walkthrough.md)

---

## Python

\`\`\`bash
cd cookbooks/code/polyglot/python
pip install -r requirements.txt
python app.py
\`\`\`

**Source:** [app.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/app.py) | [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/walkthrough.md)

---

## The Contract

All task apps implement the same [OpenAPI spec](https://github.com/synth-laboratories/synth-ai/blob/main/synth_ai/contracts/task_app.yaml):

- `GET /health` — health check
- `POST /rollout` — evaluate prompt candidate

**Details:** [polyglot-task-apps](/prompt-optimization/polyglot-task-apps)
```

---

## Navigation Update

`monorepo/docs/docs.json`:

```json
{
  "group": "Prompt Optimization",
  "pages": [
    "cookbooks/prompt-optimization-in-process",
    "cookbooks/prompt-optimization-polyglot"
  ]
}
```

Remove old stubs:
- `cookbooks/prompt-optimization-mipro.mdx`
- `cookbooks/prompt-optimization-gepa.mdx`

---

## Checklist

- [ ] Create `prompt-optimization-in-process.mdx`
- [ ] Create `prompt-optimization-polyglot.mdx`
- [ ] Update `docs.json` navigation
- [ ] Delete old stub files
- [ ] Verify all GitHub links resolve


EXAMPLES

Here are the two MDX cookbook pages plus the `docs.json` navigation snippet.

````mdx
---
// monorepo/docs/cookbooks/prompt-optimization-in-process.mdx
title: "In-Process Prompt Optimization"
description: "Run GEPA or MIPRO from a single Python script"
---

```bash
# GEPA: Banking77 (in-process)
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/gepa/run_walkthrough.py
```

**Source:** [run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/run_walkthrough.py) · [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/config.toml)

## Prerequisites

- Python 3.11+
- `uv` package manager
- API keys available as env vars or in `.env`:
  - `SYNTH_API_KEY`
  - `GROQ_API_KEY`

---

## GEPA: Banking77

Single-command run (same as above):

```bash
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/gepa/run_walkthrough.py
```

### What happens

1. Task app starts in an in-process FastAPI server
2. Cloudflare quick tunnel is created automatically
3. GEPA prompt-learning job is submitted and polled until completion
4. Results are printed and all resources are cleaned up on exit

### Results

Job `pl_5ea04259c2fd4c7a` — **83.33% accuracy** in ~35s

```json
{
  "job_id": "pl_5ea04259c2fd4c7a",
  "best_score": 0.8333,
  "algorithm": "gepa",
  "total_time_seconds": 35.6
}
```

**Full results:** [results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/results.json)

---

## MIPRO: Banking77

Run the MIPRO variant the same way:

```bash
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/mipro/run_walkthrough.py
```

**Source:** [run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/run_walkthrough.py) · [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/config.toml)

### Results

Job `pl_e95cc778c0fb4742` — **60% accuracy** in ~130s

**Full results:** [results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/results.json)

---

## Core Pattern

Minimal in-process pattern shared by both GEPA and MIPRO:

```python
import asyncio

from synth_ai.task import InProcessTaskApp
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob


async def main() -> None:
    async with InProcessTaskApp(
        task_app_path="task_app.py",
        port=8001,
    ) as task_app:
        job = PromptLearningJob.from_config(
            config_path="config.toml",
            task_app_url=task_app.url,
        )
        results = await job.poll_until_complete()
        print(results)


if __name__ == "__main__":
    asyncio.run(main())
```

**Full walkthrough:** [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/walkthrough.md)
````

````mdx
---
// monorepo/docs/cookbooks/prompt-optimization-polyglot.mdx
title: "Polyglot Task Apps"
description: "Prompt optimization with task apps in Rust, Go, TypeScript, Python"
---

```bash
# Run GEPA against any language-specific Banking77 task app
synth train gepa_config.toml --poll
```

Task apps work in any language; the optimizer only needs an HTTP endpoint.

## Results Summary

| Language   | Job ID                 | Accuracy | Source |
|-----------|------------------------|----------|--------|
| Rust       | `pl_4f69a1b099a14e4b`  | **100%** | [main.rs](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/src/main.rs) |
| TypeScript | `pl_787c47998cfe4745`  | 85.7%    | [index.ts](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/src/index.ts) |
| Python     | `pl_7e0227cc41454ec5`  | 66.7%    | [app.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/app.py) |
| Go         | `pl_1dd94dfdc8c6479d`  | 60%      | [main.go](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/main.go) |

---

## Rust

```bash
cd cookbooks/code/polyglot/rust

# 1. Build the Banking77 task app
cargo build --release

# 2. Start the task app (uses ENVIRONMENT_API_KEY for auth)
ENVIRONMENT_API_KEY=secret ./target/release/synth-task-app
```

Expose via tunnel (if you’re not using in-process mode):

```bash
cloudflared tunnel --url http://localhost:8001
```

Run GEPA prompt optimization:

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[main.rs](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/src/main.rs) · [gepa_config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/gepa_config.toml) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/walkthrough.md)

---

## TypeScript

```bash
cd cookbooks/code/polyglot/typescript

# 1. Install deps
npm install

# 2. Start the task app
ENVIRONMENT_API_KEY=secret npm run dev
```

Then, from another shell:

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[index.ts](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/src/index.ts) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/walkthrough.md)

---

## Go

```bash
cd cookbooks/code/polyglot/go

# 1. Build a static task-app binary
go build -o synth-task-app

# 2. Run the task app
ENVIRONMENT_API_KEY=secret ./synth-task-app
```

Run optimization:

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[main.go](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/main.go) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/walkthrough.md)

---

## Python

```bash
cd cookbooks/code/polyglot/python

# 1. Install deps
pip install -r requirements.txt

# 2. Run the task app
ENVIRONMENT_API_KEY=secret python app.py
```

Run optimization:

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[app.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/app.py) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/walkthrough.md)

---

## The Contract

All task apps implement the same [OpenAPI spec](https://github.com/synth-laboratories/synth-ai/blob/main/synth_ai/contracts/task_app.yaml):

- `GET /health` — health check
- `POST /rollout` — evaluate prompt candidate
- `GET /task_info` — optional dataset metadata

**Details:** [polyglot-task-apps](/prompt-optimization/polyglot-task-apps)
````

```json
{
  "group": "Prompt Optimization",
  "pages": [
    "cookbooks/prompt-optimization-in-process",
    "cookbooks/prompt-optimization-polyglot"
  ]
}
```
vHere are the two MDX cookbook pages plus the `docs.json` nav update.

````mdx
---
// monorepo/docs/cookbooks/prompt-optimization-in-process.mdx
title: "In-Process Prompt Optimization"
description: "Run GEPA or MIPRO from a single Python script"
---

```bash
# GEPA (Banking77)
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/gepa/run_walkthrough.py

# MIPRO (Banking77)
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/mipro/run_walkthrough.py
```

Sources:
- GEPA: [run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/run_walkthrough.py) · [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/config.toml) · [results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/results.json)
- MIPRO: [run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/run_walkthrough.py) · [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/config.toml) · [results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/results.json)

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager
- Accounts + API keys set up for Synth and your models
- Environment variables:
  - `SYNTH_API_KEY`
  - `ENVIRONMENT_API_KEY`
  - `GROQ_API_KEY` (or other model provider key, depending on config)

---

## GEPA: Banking77 (In-Process)

Run a full GEPA optimization loop from a single script:

```bash
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/gepa/run_walkthrough.py
```

**Source:**  
[run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/run_walkthrough.py) · [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/config.toml) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/walkthrough.md)

### What happens

1. An in-process task app is started in a background thread.
2. A Cloudflare tunnel is created automatically and registered with Synth.
3. A GEPA prompt-learning job is submitted, monitored, and polled until completion.
4. Final results are returned to Python, and the task app + tunnel are cleaned up automatically.

### Sample results

Job `pl_5ea04259c2fd4c7a` — **83.33% accuracy** on Banking77 in ~35 seconds.

```json
{
  "job_id": "pl_5ea04259c2fd4c7a",
  "algorithm": "gepa",
  "dataset": "banking77",
  "best_score": 0.8333,
  "best_prompt_rank": 1,
  "num_generations": 8,
  "total_time_seconds": 35.6
}
```

See full run output in:  
[results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/results.json)

---

## MIPRO: Banking77 (In-Process)

Swap the algorithm to MIPRO, keep the same in-process pattern:

```bash
cd synth-ai
uv run python /path/to/cookbooks/code/training/prompt_learning/mipro/run_walkthrough.py
```

**Source:**  
[run_walkthrough.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/run_walkthrough.py) · [config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/config.toml) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/walkthrough.md)

### Sample results

Job `pl_e95cc778c0fb4742` — **60.0% accuracy** on Banking77 in ~130 seconds.

```json
{
  "job_id": "pl_e95cc778c0fb4742",
  "algorithm": "mipro",
  "dataset": "banking77",
  "best_score": 0.60,
  "best_prompt_rank": 1,
  "total_time_seconds": 130.4
}
```

See full run output in:  
[results.json](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/results.json)

---

## Core pattern (GEPA + MIPRO)

Both walkthroughs share the same in-process pattern: start a task app, create a tunnel, run a prompt-learning job pointed at the tunnel URL, and clean everything up when done.

```python
from synth_ai.task import InProcessTaskApp
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob


async def optimize_prompt(
    task_app_path: str,
    config_path: str,
    port: int = 8001,
):
    async with InProcessTaskApp(
        task_app_path=task_app_path,
        port=port,
    ) as task_app:
        # Build job from TOML config
        job = PromptLearningJob.from_config(
            config_path=config_path,
            task_app_url=task_app.url,
        )

        # Run until complete and fetch best prompt + metrics
        results = await job.poll_until_complete()

    return results
```

Minimal GEPA/MIPRO wrappers:

```python
# GEPA
results = await optimize_prompt(
    task_app_path="task_app.py",
    config_path="code/training/prompt_learning/gepa/config.toml",
)

# MIPRO
results = await optimize_prompt(
    task_app_path="task_app.py",
    config_path="code/training/prompt_learning/mipro/config.toml",
)
```

Full GEPA walkthrough:  
[gepa/walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/gepa/walkthrough.md)

Full MIPRO walkthrough:  
[mipro/walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/training/prompt_learning/mipro/walkthrough.md)
````

---

````mdx
---
// monorepo/docs/cookbooks/prompt-optimization-polyglot.mdx
title: "Polyglot Task Apps"
description: "Prompt optimization with task apps in Rust, Go, TypeScript, and Python"
---

```bash
# Run GEPA prompt optimization against any task app
# (after starting your task app and tunnel)
synth train gepa_config.toml --poll
```

Task apps can be written in any language as long as they implement the Synth task app contract (`/health` and `/rollout`).

## Results summary (Banking77, GEPA)

| Language   | Job ID                 | Accuracy | Source                                                                 |
|-----------:|------------------------|---------:|------------------------------------------------------------------------|
| Rust       | `pl_4f69a1b099a14e4b`  | **100%** | [main.rs](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/src/main.rs) |
| TypeScript | `pl_787c47998cfe4745`  | 85.7%    | [index.ts](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/src/index.ts) |
| Python     | `pl_7e0227cc41454ec5`  | 66.7%    | [app.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/app.py) |
| Go         | `pl_1dd94dfdc8c6479d`  | 60.0%    | [main.go](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/main.go) |

All jobs optimize prompts for the same Banking77 task app; only the implementation language changes.

---

## Prerequisites

- Synth CLI installed (`synth` on your PATH)
- Task app contract implemented in one of:
  - Rust
  - TypeScript (Node.js / Bun / Deno)
  - Python
  - Go
- `ENVIRONMENT_API_KEY` set in your shell
- A Cloudflare tunnel or other public URL exposed for `http://localhost:8001`

---

## Rust task app

```bash
cd cookbooks/code/polyglot/rust

# Build release binary
cargo build --release

# Start task app on localhost:8001
ENVIRONMENT_API_KEY=secret ./target/release/synth-task-app
```

Expose it via Cloudflare:

```bash
cloudflared tunnel --url http://localhost:8001
```

Run GEPA optimization (Banking77):

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[main.rs](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/src/main.rs) · [gepa_config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/gepa_config.toml) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/rust/walkthrough.md)

### Sample result (Rust)

Job `pl_4f69a1b099a14e4b` — **100% accuracy** on Banking77.

```json
{
  "job_id": "pl_4f69a1b099a14e4b",
  "language": "rust",
  "algorithm": "gepa",
  "best_score": 1.0,
  "total_time_seconds": 42.1
}
```

---

## TypeScript task app

```bash
cd cookbooks/code/polyglot/typescript

# Install dependencies and start the server on localhost:8001
npm install
ENVIRONMENT_API_KEY=secret npm run dev
```

Expose it via Cloudflare:

```bash
cloudflared tunnel --url http://localhost:8001
```

Run GEPA optimization:

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[index.ts](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/src/index.ts) · [gepa_config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/gepa_config.toml) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/typescript/walkthrough.md)

### Sample result (TypeScript)

Job `pl_787c47998cfe4745` — **85.7% accuracy** on Banking77.

```json
{
  "job_id": "pl_787c47998cfe4745",
  "language": "typescript",
  "algorithm": "gepa",
  "best_score": 0.8571
}
```

---

## Go task app

```bash
cd cookbooks/code/polyglot/go

# Build and run binary on localhost:8001
go build -o synth-task-app
ENVIRONMENT_API_KEY=secret ./synth-task-app
```

Expose via Cloudflare:

```bash
cloudflared tunnel --url http://localhost:8001
```

Run GEPA optimization:

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[main.go](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/main.go) · [gepa_config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/gepa_config.toml) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/go/walkthrough.md)

### Sample result (Go)

Job `pl_1dd94dfdc8c6479d` — **60.0% accuracy** on Banking77.

```json
{
  "job_id": "pl_1dd94dfdc8c6479d",
  "language": "go",
  "algorithm": "gepa",
  "best_score": 0.60
}
```

---

## Python task app

```bash
cd cookbooks/code/polyglot/python

# Install dependencies and start FastAPI/Flask app on localhost:8001
pip install -r requirements.txt
ENVIRONMENT_API_KEY=secret python app.py
```

Expose via Cloudflare:

```bash
cloudflared tunnel --url http://localhost:8001
```

Run GEPA optimization:

```bash
synth train gepa_config.toml --poll
```

**Source:**  
[app.py](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/app.py) · [gepa_config.toml](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/gepa_config.toml) · [walkthrough.md](https://github.com/synth-laboratories/cookbooks/blob/main/code/polyglot/python/walkthrough.md)

### Sample result (Python)

Job `pl_7e0227cc41454ec5` — **66.7% accuracy** on Banking77.

```json
{
  "job_id": "pl_7e0227cc41454ec5",
  "language": "python",
  "algorithm": "gepa",
  "best_score": 0.6667
}
```

---

## The task app contract

All of these task apps implement the same HTTP contract:

- `GET /health` — basic health check used by Synth to discover and monitor the app
- `POST /rollout` — evaluate a batch of prompt candidates and return rewards/metrics

OpenAPI spec (for all languages):

- GitHub: [task_app.yaml](https://github.com/synth-laboratories/synth-ai/blob/main/synth_ai/contracts/task_app.yaml)
- Raw: `https://raw.githubusercontent.com/synth-laboratories/synth-ai/main/synth_ai/contracts/task_app.yaml`

Key rules:

- `ENVIRONMENT_API_KEY` must match the `X-API-Key` header on `/rollout` requests.
- Your `/rollout` handler:
  - Reads the prompt candidate from the request payload.
  - Runs it against your task (e.g., Banking77 classifier).
  - Returns per-sample rewards and aggregate metrics in the response.

See also: [polyglot walkthroughs in the cookbooks repo](https://github.com/synth-laboratories/cookbooks/tree/main/code/polyglot).
````

---

```json
// monorepo/docs/docs.json (navigation fragment)
{
  "group": "Prompt Optimization",
  "pages": [
    "cookbooks/prompt-optimization-in-process",
    "cookbooks/prompt-optimization-polyglot"
  ]
}
```
