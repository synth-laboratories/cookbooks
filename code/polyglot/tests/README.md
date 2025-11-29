# Polyglot Cookbook Tests

Smoke scripts to validate the Banking77 polyglot Task App end-to-end against the Synth backend (prod) once your Task App is running and exposed via HTTPS (tunnel or deployment).

## Prerequisites
- Task App running locally (Rust/Go/TS/Zig) with `ENVIRONMENT_API_KEY` set.
- HTTPS URL for the Task App (from `cloudflared` or your deployment).
- Env vars:
  - `SYNTH_API_KEY` — Synth auth
  - `TASK_APP_URL` — public URL to your Task App (e.g., `https://<tunnel>.trycloudflare.com`)
  - `TASK_APP_API_KEY` — must match the Task App's `ENVIRONMENT_API_KEY`
  - `SYNTH_API_URL` (optional, default `https://agent-learning.onrender.com`)
  - `SYNTH_ALGORITHM` (optional, default `mipro`; set to `gepa` to test GEPA)

## Scripts
- `run_smoke.sh` — verifies `/health`, `/task_info`, `/rollout`, then submits a Synth job (MIPRO or GEPA) to prod.

Usage:
```bash
cd cookbooks/code/training/polyglot/tests
chmod +x run_smoke.sh
./run_smoke.sh
```

The script exits non-zero on failure so you can integrate it into CI. It does not start the Task App; start it separately before running.
