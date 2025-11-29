# Cookbook Verification Checklist

This document tracks verification of all cookbook examples. Run each command and record the results.

## Prerequisites

```bash
# Required environment variables
export SYNTH_API_KEY="your-synth-api-key"
export OPENAI_API_KEY="your-openai-key"
export ENVIRONMENT_API_KEY="test-env-key-123"

# Required tools
python --version   # 3.11+
uvx --version      # uv tool runner
cloudflared --version  # tunnel (optional but recommended)
```

---

## 1. Polyglot Task Apps

### 1.1 Python Task App

```bash
cd code/polyglot/python
pip install -r requirements.txt
python app.py
```

**Expected output:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

**Verification (in another terminal):**
```bash
# Health check (no auth)
curl http://localhost:8001/health
# Expected: {"status":"ok"}

# Task info (with auth)
curl -H "X-API-Key: test-env-key-123" http://localhost:8001/task_info
# Expected: {"name":"banking77","description":"Intent classification...","count":100}

# Rollout (with auth)
curl -X POST http://localhost:8001/rollout \
  -H "X-API-Key: test-env-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "env": {"seed": 0},
    "policy": {
      "config": {
        "prompt_template": "Classify the banking intent: {{text}}",
        "inference_url": "https://api.openai.com/v1"
      }
    }
  }'
# Expected: {"metrics":{"mean_return":0.0 or 1.0},"trajectories":[...]}
```

- [ ] Health endpoint works
- [ ] Task info returns dataset metadata
- [ ] Rollout calls LLM and returns reward

**Actual Results:**
```
# Paste actual output here
```

---

### 1.2 Go Task App

```bash
cd code/polyglot/go
go build -o synth-task-app
./synth-task-app
```

**Expected output:**
```
Starting Banking77 Task App on :8001
```

**Verification:**
```bash
curl http://localhost:8001/health
curl -H "X-API-Key: test-env-key-123" http://localhost:8001/task_info
curl -X POST http://localhost:8001/rollout \
  -H "X-API-Key: test-env-key-123" \
  -H "Content-Type: application/json" \
  -d '{"env":{"seed":0},"policy":{"config":{"prompt_template":"Classify: {{text}}","inference_url":"https://api.openai.com/v1"}}}'
```

- [ ] Health endpoint works
- [ ] Task info returns dataset metadata
- [ ] Rollout calls LLM and returns reward

**Actual Results:**
```
# Paste actual output here
```

---

### 1.3 Rust Task App

```bash
cd code/polyglot/rust
cargo build --release
./target/release/synth-task-app
```

**Expected output:**
```
Starting server on 0.0.0.0:8001
```

**Verification:**
```bash
curl http://localhost:8001/health
curl -H "X-API-Key: test-env-key-123" http://localhost:8001/task_info
curl -X POST http://localhost:8001/rollout \
  -H "X-API-Key: test-env-key-123" \
  -H "Content-Type: application/json" \
  -d '{"env":{"seed":0},"policy":{"config":{"prompt_template":"Classify: {{text}}","inference_url":"https://api.openai.com/v1"}}}'
```

- [ ] Health endpoint works
- [ ] Task info returns dataset metadata
- [ ] Rollout calls LLM and returns reward

**Actual Results:**
```
# Paste actual output here
```

---

### 1.4 TypeScript Task App

```bash
cd code/polyglot/typescript
npm install
npm run dev
```

**Expected output:**
```
Server running on http://localhost:8001
```

**Verification:**
```bash
curl http://localhost:8001/health
curl -H "X-API-Key: test-env-key-123" http://localhost:8001/task_info
curl -X POST http://localhost:8001/rollout \
  -H "X-API-Key: test-env-key-123" \
  -H "Content-Type: application/json" \
  -d '{"env":{"seed":0},"policy":{"config":{"prompt_template":"Classify: {{text}}","inference_url":"https://api.openai.com/v1"}}}'
```

- [ ] Health endpoint works
- [ ] Task info returns dataset metadata
- [ ] Rollout calls LLM and returns reward

**Actual Results:**
```
# Paste actual output here
```

---

### 1.5 Zig Task App

```bash
cd code/polyglot/zig
zig build -Doptimize=ReleaseFast
./zig-out/bin/synth-task-app
```

**Expected output:**
```
Listening on 0.0.0.0:8001
```

**Verification:**
```bash
curl http://localhost:8001/health
curl -H "X-API-Key: test-env-key-123" http://localhost:8001/task_info
curl -X POST http://localhost:8001/rollout \
  -H "X-API-Key: test-env-key-123" \
  -H "Content-Type: application/json" \
  -d '{"env":{"seed":0},"policy":{"config":{"prompt_template":"Classify: {{text}}","inference_url":"https://api.openai.com/v1"}}}'
```

- [ ] Health endpoint works
- [ ] Task info returns dataset metadata
- [ ] Rollout calls LLM and returns reward

**Actual Results:**
```
# Paste actual output here
```

---

## 2. Prompt Learning - MIPRO

### 2.1 Task App Setup

```bash
cd code/training/prompt_learning
pip install fastapi uvicorn datasets httpx
python task_app.py
```

**Expected output:**
```
Loading Banking77 from HuggingFace...
Dataset loaded: 3080 examples
Starting Banking77 Task App on port 8001
```

- [ ] Dataset loads from HuggingFace
- [ ] Server starts on port 8001

---

### 2.2 MIPRO Interactive Walkthrough

```bash
cd code/training/prompt_learning/mipro
bash run_interactive.sh
```

**Expected flow:**
1. Prerequisites check passes
2. ENVIRONMENT_API_KEY generated
3. Task app starts
4. MIPRO config created
5. Job submitted and runs
6. Results displayed

- [ ] Prerequisites check passes
- [ ] Task app starts successfully
- [ ] MIPRO job completes
- [ ] Best prompt returned

**Actual Results:**
```
# Paste actual output here
```

---

### 2.3 MIPRO SDK Usage

```bash
cd code/training/prompt_learning
python sdk/basic.py
```

**Expected output:**
```
Starting MIPRO optimization...
Job ID: mipro_xxx
Status: completed
Best score: 0.XX
```

- [ ] SDK connects to backend
- [ ] Job runs successfully
- [ ] Results returned

**Actual Results:**
```
# Paste actual output here
```

---

## 3. Prompt Learning - GEPA

### 3.1 GEPA Interactive Walkthrough

```bash
cd code/training/prompt_learning/gepa
bash run_interactive.sh
```

**Expected flow:**
1. Prerequisites check passes
2. ENVIRONMENT_API_KEY generated
3. Task app starts
4. GEPA config created
5. Job submitted and runs
6. Results displayed

- [ ] Prerequisites check passes
- [ ] Task app starts successfully
- [ ] GEPA job completes
- [ ] Best prompt returned

**Actual Results:**
```
# Paste actual output here
```

---

### 3.2 GEPA In-Process Mode

```bash
cd code/training/prompt_learning/gepa
python run_gepa_inprocess.py
```

**Expected output:**
```
Starting in-process GEPA optimization...
Task app running at http://localhost:XXXX
Generation 1/3...
...
Best score: 0.XX
```

- [ ] In-process task app starts
- [ ] GEPA runs locally
- [ ] Results returned

**Actual Results:**
```
# Paste actual output here
```

---

## 4. End-to-End Optimization Test

This test runs a full optimization cycle with a polyglot task app.

### 4.1 Start Task App + Tunnel

```bash
# Terminal 1: Start Python task app
cd code/polyglot/python
python app.py

# Terminal 2: Start tunnel
cloudflared tunnel --url http://localhost:8001
# Note the tunnel URL: https://xxx.trycloudflare.com
```

### 4.2 Submit MIPRO Job

```bash
curl -X POST https://synth-laboratories-prod--learning-v2-service-fastapi-app.modal.run/api/prompt-learning/online/jobs \
  -H "Authorization: Bearer $SYNTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "mipro",
    "config_body": {
      "prompt_learning": {
        "task_app_url": "https://YOUR-TUNNEL-URL",
        "task_app_api_key": "test-env-key-123"
      },
      "termination": {
        "budget": 20
      }
    }
  }'
```

**Expected response:**
```json
{
  "job_id": "mipro_xxx",
  "status": "queued"
}
```

- [ ] Job submitted successfully
- [ ] Job ID returned
- [ ] Job completes (poll status)

**Actual Results:**
```
# Paste job_id and final status here
```

---

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Python Task App | ⬜ | |
| Go Task App | ⬜ | |
| Rust Task App | ⬜ | |
| TypeScript Task App | ⬜ | |
| Zig Task App | ⬜ | |
| MIPRO Interactive | ⬜ | |
| MIPRO SDK | ⬜ | |
| GEPA Interactive | ⬜ | |
| GEPA In-Process | ⬜ | |
| E2E Optimization | ⬜ | |

**Legend:** ✅ Verified | ⬜ Pending | ❌ Failed
