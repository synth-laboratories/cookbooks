#!/bin/bash
# MIPRO Prompt Optimization via CLI
#
# This script runs MIPRO prompt optimization using the CLI.
#
# Prerequisites:
#   - Task app running: python ../task_app.py
#   - Environment variables: SYNTH_API_KEY, ENVIRONMENT_API_KEY
#
# Usage:
#   bash run_mipro.sh
#
# Options:
#   - Edit configs/mipro.toml to change optimization parameters
#   - Use --task-url to override task app URL
#   - Use --backend to override backend URL

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment if .env exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment from .env"
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Verify required environment variables
if [ -z "$SYNTH_API_KEY" ]; then
    echo "Error: SYNTH_API_KEY not set"
    echo "Set it with: export SYNTH_API_KEY=your-key"
    exit 1
fi

if [ -z "$ENVIRONMENT_API_KEY" ]; then
    echo "Error: ENVIRONMENT_API_KEY not set"
    echo "Set it with: export ENVIRONMENT_API_KEY=your-env-key"
    exit 1
fi

# Config path
CONFIG_PATH="$COOKBOOK_DIR/configs/mipro.toml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

echo "========================================"
echo "MIPRO Prompt Optimization"
echo "========================================"
echo "Config: $CONFIG_PATH"
echo "API Key: ${SYNTH_API_KEY:0:8}..."
echo "========================================"

# Run training
uvx synth-ai train "$CONFIG_PATH" \
    --poll \
    --poll-timeout 3600 \
    --stream-format cli \
    "$@"

echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "Check results/ directory for output files"
