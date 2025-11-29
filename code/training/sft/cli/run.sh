#!/bin/bash
# Banking77 SFT Training via CLI
#
# This script runs supervised fine-tuning on Banking77 intent classification.
#
# Prerequisites:
#   - Environment variables: SYNTH_API_KEY
#
# Usage:
#   bash run.sh
#
# Options:
#   - Edit configs/sft.toml to change training parameters
#   - Use --examples to limit training data (e.g., --examples 100)

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

# Config path
CONFIG_PATH="$COOKBOOK_DIR/configs/sft.toml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

echo "========================================"
echo "Banking77 SFT Training"
echo "========================================"
echo "Config: $CONFIG_PATH"
echo "API Key: ${SYNTH_API_KEY:0:8}..."
echo "========================================"

# Run training
uvx synth-ai train "$CONFIG_PATH" \
    --poll \
    --poll-timeout 7200 \
    --stream-format chart \
    "$@"

echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "Check results/ directory for model info"
