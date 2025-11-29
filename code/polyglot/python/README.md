# Python Task App (FastAPI)

Minimal FastAPI implementation of the Banking77 Task App. It exposes the required endpoints (`/health`, `/task_info`, `/rollout`), calls an LLM (OpenAI/Groq) using `inference_url`, and computes rewards against `data/banking77.json`.

## Prerequisites
- Python 3.11+
- Env vars:
  - `ENVIRONMENT_API_KEY` — inbound auth (`X-API-Key`)
  - `OPENAI_API_KEY` or `GROQ_API_KEY` — outbound LLM
  - `PORT` (optional, default `8001`)

Install deps:
```bash
cd cookbooks/code/training/polyglot/python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run:
```bash
ENVIRONMENT_API_KEY=demo OPENAI_API_KEY=sk-... uvicorn app:app --host 0.0.0.0 --port 8001
```

Smoke:
```bash
curl http://localhost:8001/health
curl -H "X-API-Key: demo" http://localhost:8001/task_info
curl -H "X-API-Key: demo" -H "Content-Type: application/json" \
  -d '{"env":{"seed":0},"policy":{"config":{"prompt_template":"Classify {{text}}","inference_url":"https://api.openai.com/v1?model=gpt-4o-mini"}}}' \
  http://localhost:8001/rollout
```
