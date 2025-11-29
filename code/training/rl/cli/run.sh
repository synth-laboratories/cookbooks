#!/bin/bash
# RL Training via CLI
#
# This script demonstrates the CLI workflow for RL training:
# 1. Load environment variables
# 2. Start task app (or use existing one)
# 3. Submit training job
# 4. Poll until complete
#
# Usage:
#   ./run.sh              # Full workflow
#   ./run.sh --no-poll    # Submit without polling
#   ./run.sh --task-only  # Just start task app

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment from .env..."
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
elif [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading environment from project .env..."
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Check required environment variables
if [ -z "$SYNTH_API_KEY" ]; then
    echo "Error: SYNTH_API_KEY not set"
    echo "Create .env file with: SYNTH_API_KEY=your-key"
    exit 1
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo "Error: ENVIRONMENT_API_KEY not set"
    echo "Create .env file with: ENVIRONMENT_API_KEY=your-env-key"
    exit 1
fi

# Configuration
CONFIG_PATH="$PROJECT_DIR/configs/rl.toml"
TASK_APP_PORT="${TASK_APP_PORT:-8114}"

# Parse arguments
POLL=true
TASK_ONLY=false
START_TASK_APP=true

for arg in "$@"; do
    case $arg in
        --no-poll)
            POLL=false
            ;;
        --task-only)
            TASK_ONLY=true
            ;;
        --no-task-app)
            START_TASK_APP=false
            ;;
    esac
done

# Function to start task app
start_task_app() {
    echo "Starting heart disease task app on port $TASK_APP_PORT..."

    # Check if already running
    if curl -s "http://localhost:$TASK_APP_PORT/health" > /dev/null 2>&1; then
        echo "Task app already running on port $TASK_APP_PORT"
        return 0
    fi

    # Start task app in background
    python -m synth_ai.sdk.task.server \
        --app heartdisease \
        --port "$TASK_APP_PORT" \
        --env-key "$ENVIRONMENT_API_KEY" &

    TASK_APP_PID=$!
    echo "Task app started (PID: $TASK_APP_PID)"

    # Wait for it to be ready
    echo "Waiting for task app to be ready..."
    for i in {1..30}; do
        if curl -s "http://localhost:$TASK_APP_PORT/health" > /dev/null 2>&1; then
            echo "Task app ready!"
            return 0
        fi
        sleep 1
    done

    echo "Error: Task app failed to start"
    kill $TASK_APP_PID 2>/dev/null || true
    exit 1
}

# Function to stop task app
stop_task_app() {
    if [ -n "$TASK_APP_PID" ]; then
        echo "Stopping task app (PID: $TASK_APP_PID)..."
        kill $TASK_APP_PID 2>/dev/null || true
    fi
}

# Cleanup on exit
trap stop_task_app EXIT

# Task app only mode
if [ "$TASK_ONLY" = true ]; then
    echo "Starting task app only mode..."
    start_task_app
    echo "Task app running. Press Ctrl+C to stop."
    wait $TASK_APP_PID
    exit 0
fi

# Start task app if needed
if [ "$START_TASK_APP" = true ]; then
    start_task_app
fi

# Submit training job
echo ""
echo "=========================================="
echo "Submitting RL Training Job"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo ""

if [ "$POLL" = true ]; then
    # Submit and poll
    uvx synth-ai train "$CONFIG_PATH" \
        --poll \
        --timeout 7200
else
    # Submit only
    uvx synth-ai train "$CONFIG_PATH"
fi

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
