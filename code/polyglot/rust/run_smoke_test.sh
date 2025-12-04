#!/usr/bin/env bash
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║   GEPA Smoke Test Runner for Rust Task App                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Usage:
#   ./run_smoke_test.sh
#
# Prerequisites:
#   - SYNTH_API_KEY and ENVIRONMENT_API_KEY set (or source synth-ai/.env)
#   - cloudflared installed for tunnel
#   - cargo installed for building Rust

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# synth-ai is a sibling of cookbooks
SYNTH_AI_ENV="${SCRIPT_DIR}/../../../../synth-ai/.env"

# ─────────────────────────────────────────────────────────────────────────────
# Load environment
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f "$SYNTH_AI_ENV" ]]; then
    echo "✓ Loading .env from synth-ai"
    set -a
    source "$SYNTH_AI_ENV"
    set +a
fi

if [[ -z "${SYNTH_API_KEY:-}" ]]; then
    echo "❌ SYNTH_API_KEY is required"
    exit 1
fi

if [[ -z "${ENVIRONMENT_API_KEY:-}" ]]; then
    echo "❌ ENVIRONMENT_API_KEY is required"
    exit 1
fi

BACKEND_URL="${BACKEND_URL:-https://agent-learning.onrender.com/api}"
PORT="${PORT:-8001}"

echo "
╭───────────────────────────────────────────────────────────────────────────╮
│  🦀 GEPA Smoke Test for Rust                                              │
╰───────────────────────────────────────────────────────────────────────────╯
  Backend URL:     $BACKEND_URL
  Task App Port:   $PORT
  Config:          $SCRIPT_DIR/gepa_smoke_test.toml
"

# ─────────────────────────────────────────────────────────────────────────────
# Build and start task app
# ─────────────────────────────────────────────────────────────────────────────
echo "📦 Building Rust task app..."
cd "$SCRIPT_DIR"
cargo build --release 2>&1 | tail -5

# Check if task app is already running
if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "✓ Task app already running on port $PORT"
else
    echo "🚀 Starting task app on port $PORT..."
    ENVIRONMENT_API_KEY="$ENVIRONMENT_API_KEY" PORT="$PORT" \
        ./target/release/synth-task-app &
    TASK_APP_PID=$!
    sleep 2
    
    if ! curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "❌ Task app failed to start"
        exit 1
    fi
    echo "✓ Task app started (PID: $TASK_APP_PID)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Create tunnel (if backend is remote)
# ─────────────────────────────────────────────────────────────────────────────
TASK_APP_URL="http://localhost:$PORT"

if [[ "$BACKEND_URL" == *"onrender.com"* ]] || [[ "$BACKEND_URL" == *"https://"* ]]; then
    echo "🌐 Creating Cloudflare quick tunnel..."
    
    TUNNEL_LOG="/tmp/cloudflared_rust_$PORT.log"
    cloudflared tunnel --url "http://localhost:$PORT" > "$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    
    # Wait for tunnel URL
    for i in {1..30}; do
        TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
        if [[ -n "$TUNNEL_URL" ]]; then
            break
        fi
        sleep 1
    done
    
    if [[ -z "${TUNNEL_URL:-}" ]]; then
        echo "❌ Failed to create tunnel"
        cat "$TUNNEL_LOG"
        exit 1
    fi
    
    echo "✅ Tunnel created: $TUNNEL_URL"
    echo "   PID: $TUNNEL_PID"
    
    # Wait for DNS propagation
    echo "⏳ Waiting for tunnel DNS (5s)..."
    sleep 5
    
    TASK_APP_URL="$TUNNEL_URL"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Submit GEPA job
# ─────────────────────────────────────────────────────────────────────────────
echo "📤 Submitting GEPA job..."
echo "   Task App URL: $TASK_APP_URL"

# Read and process config
CONFIG_CONTENT=$(cat "$SCRIPT_DIR/gepa_smoke_test.toml" | \
    sed "s|\\\${TASK_APP_URL}|$TASK_APP_URL|g" | \
    sed "s|\\\${TASK_APP_API_KEY}|$ENVIRONMENT_API_KEY|g")

# Submit job
RESPONSE=$(curl -s -X POST "$BACKEND_URL/prompt-learning/online/jobs" \
    -H "Authorization: Bearer $SYNTH_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"algorithm\": \"gepa\",
        \"config_body\": \"$(echo "$CONFIG_CONTENT" | sed 's/"/\\"/g' | tr '\n' ' ')\"
    }")

JOB_ID=$(echo "$RESPONSE" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [[ -z "$JOB_ID" ]]; then
    echo "❌ Job submission failed:"
    echo "$RESPONSE" | head -20
    exit 1
fi

echo "✅ Job submitted: $JOB_ID"

# ─────────────────────────────────────────────────────────────────────────────
# Poll for completion
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "⏳ Polling for completion (this may take 2-5 minutes)..."

for i in {1..120}; do
    STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $SYNTH_API_KEY" \
        "$BACKEND_URL/prompt-learning/online/jobs/$JOB_ID")
    
    STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    
    if [[ "$STATUS" == "succeeded" ]] || [[ "$STATUS" == "completed" ]]; then
        echo ""
        echo "✅ Job completed successfully!"
        
        BEST_SCORE=$(echo "$STATUS_RESPONSE" | grep -o '"prompt_best_score":[0-9.]*' | cut -d':' -f2)
        echo "   Best Score: ${BEST_SCORE:-N/A}"
        
        break
    elif [[ "$STATUS" == "failed" ]]; then
        echo ""
        echo "❌ Job failed!"
        echo "$STATUS_RESPONSE" | head -30
        exit 1
    fi
    
    # Progress indicator
    printf "\r   [%3d/120] Status: %-12s" "$i" "$STATUS"
    sleep 5
done

echo ""
echo "🎉 Rust GEPA smoke test complete!"

# Cleanup
if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
fi
if [[ -n "${TASK_APP_PID:-}" ]]; then
    kill "$TASK_APP_PID" 2>/dev/null || true
fi

