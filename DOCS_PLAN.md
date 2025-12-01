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
