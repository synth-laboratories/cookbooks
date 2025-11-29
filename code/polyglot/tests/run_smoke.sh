#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the polyglot Task App:
# 1) /health
# 2) /task_info
# 3) /rollout (seed=0)
# 4) Submit a Synth job (MIPRO or GEPA) to prod
#
# Required env:
#   SYNTH_API_KEY       - Synth auth token
#   TASK_APP_URL        - Public HTTPS URL to your Task App (tunnel or deploy)
#   TASK_APP_API_KEY    - Must match ENVIRONMENT_API_KEY in the Task App
# Optional:
#   SYNTH_API_URL       - Defaults to https://agent-learning.onrender.com
#   SYNTH_ALGORITHM     - Defaults to mipro (set to gepa to test GEPA)
#   SYNTH_MODEL         - Defaults to gpt-4o-mini (controls inference_url model)

SYNTH_API_URL="${SYNTH_API_URL:-https://agent-learning.onrender.com}"
SYNTH_ALGORITHM="${SYNTH_ALGORITHM:-mipro}"
SYNTH_MODEL="${SYNTH_MODEL:-gpt-4o-mini}"

required=(SYNTH_API_KEY TASK_APP_URL TASK_APP_API_KEY)
for var in "${required[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required env: $var" >&2
    exit 1
  fi
done

echo "Smoke test starting:"
echo "  SYNTH_API_URL     = ${SYNTH_API_URL}"
echo "  SYNTH_ALGORITHM   = ${SYNTH_ALGORITHM}"
echo "  SYNTH_MODEL       = ${SYNTH_MODEL}"
echo "  TASK_APP_URL      = ${TASK_APP_URL}"

header_auth=(-H "Authorization: Bearer ${SYNTH_API_KEY}")
header_json=(-H "Content-Type: application/json")
header_app_key=(-H "X-API-Key: ${TASK_APP_API_KEY}")

step() { echo "[$(date -u +%H:%M:%S)] $*"; }

step "1) GET /health"
curl -fsSL "${TASK_APP_URL}/health" >/dev/null

step "2) GET /task_info"
curl -fsSL "${TASK_APP_URL}/task_info" "${header_app_key[@]}" >/dev/null

step "3) POST /rollout (seed=0)"
rollout_body=$(cat <<'JSON'
{
  "env": { "seed": 0 },
  "policy": {
    "config": {
      "prompt_template": "Classify the intent: {{text}}",
      "inference_url": "INFERENCE_URL_PLACEHOLDER"
    }
  }
}
JSON
)
inference_url="https://api.openai.com/v1?model=${SYNTH_MODEL}"
rollout_body="${rollout_body/INFERENCE_URL_PLACEHOLDER/${inference_url}}"
curl -fsSL "${TASK_APP_URL}/rollout" \
  "${header_app_key[@]}" \
  "${header_json[@]}" \
  -d "${rollout_body}" >/dev/null

step "4) Submit Synth job (${SYNTH_ALGORITHM})"
job_body=$(cat <<'JSON'
{
  "algorithm": "ALGO_PLACEHOLDER",
  "config_body": {
    "prompt_learning": {
      "task_app_url": "TASK_APP_URL_PLACEHOLDER",
      "task_app_api_key": "TASK_APP_API_KEY_PLACEHOLDER"
    }
  }
}
JSON
)
job_body="${job_body/ALGO_PLACEHOLDER/${SYNTH_ALGORITHM}}"
job_body="${job_body/TASK_APP_URL_PLACEHOLDER/${TASK_APP_URL}}"
job_body="${job_body/TASK_APP_API_KEY_PLACEHOLDER/${TASK_APP_API_KEY}}"

curl -fsSL "${SYNTH_API_URL}/api/prompt-learning/online/jobs" \
  "${header_auth[@]}" \
  "${header_json[@]}" \
  -d "${job_body}" >/dev/null

step "✅ Smoke test completed"
