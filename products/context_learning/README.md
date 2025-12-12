# Context Learning (Alpha) — Codex/CTF Example

Context Learning (aka Context Engineering) is an **infra training job** that optimizes the environment scripts used by terminal/coding agents. It’s especially useful for Codex‑style CTF tasks where getting the right tools, repo state, and runtime dependencies in place is half the battle.

**Status:** Alpha only. Your organization must be on the `alpha` access tier to create or stream Context Learning jobs. If you aren’t sure, contact Synth support.

## What this cookbook shows

- How to run Context Learning on a Codex/CTF task app.
- How to stream progress via SSE (no polling).
- How to download the best preflight script after training.
- How to run an end-to-end XBOW CTF example locally.

## Prereqs

- `synth-ai` SDK installed.
- `SYNTH_API_KEY` set in your environment (alpha tier).
- A compatible task app URL for your Codex/CTF environment (hosted or tunneled).
  - Export it as `CTX_LEARNING_TASK_APP_URL`.
- If your task app requires authentication, set `ENVIRONMENT_API_KEY`.

## 1. Create a config

Start from `ctf_context_learning_alpha.toml` in this folder. Replace placeholders with your task app URL and, optionally, baseline scripts.

## 2. Submit + stream (CLI)

```bash
synth-ai train --type context_learning --config ctf_context_learning_alpha.toml
```

This command:

1. Creates a Context Learning job.
2. Streams events/metrics over SSE until completion.
3. Prints the best preflight script if the job succeeds.

## 3. Submit + stream (SDK)

```python
from synth_ai.sdk.api.train import ContextLearningJob

job = ContextLearningJob.from_config("ctf_context_learning_alpha.toml")
job.submit()
final = job.stream_until_complete()
best = job.download_best_script()

print(best.preflight_script)
```

## 4. End-to-end XBOW CTF example (local)

If you have the XBOW benchmarks checked out in this repo, you can run the full
Codex/CTF loop locally. This script will start the XBOW task app, submit a tiny
Context Learning job, stream it, and print the best script.

```bash
uv run python cookbooks/products/context_learning/test_xbow_context_learning_alpha.py
```

Optional environment:

- `BACKEND_BASE_URL` to point at dev/staging.
- `XBOW_BENCHMARKS_PATH` to override benchmark location.
- `XBOW_TASK_APP_PORT` to force a port.

## Notes

- We recommend keeping baseline preflight scripts minimal and letting Context Learning discover missing tools/packages.
- You can add a postflight baseline too, but current GEPA‑based Context Learning primarily optimizes preflight.
