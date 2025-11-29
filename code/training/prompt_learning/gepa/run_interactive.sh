#!/bin/bash
# Interactive GEPA Prompt Optimization Walkthrough
#
# This script guides you through running GEPA (Genetic Evolution of Prompt Architectures)
# optimization on the Banking77 intent classification task using gpt-oss-20b.
#
# GEPA uses genetic algorithms to evolve prompt structures:
# - Mutation operators (add/remove/modify prompt sections)
# - Crossover (combine best parts of different prompts)
# - Selection (keep top-performing variants)
#
# Usage:
#   bash run_interactive.sh
#
# Prerequisites:
#   - SYNTH_API_KEY environment variable
#   - OPENAI_API_KEY or GROQ_API_KEY environment variable (for LLM inference)
#   - Python 3.11+ with synth-ai installed
#
# This script will:
#   1. Set up environment and verify prerequisites
#   2. Generate ENVIRONMENT_API_KEY for task app authentication
#   3. Start the Banking77 task app
#   4. Submit GEPA optimization job
#   5. Poll for completion and show results

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(dirname "$SCRIPT_DIR")"
WORKING_DIR="/tmp/gepa_walkthrough"

# Function to display an educational prompt and wait for user
prompt_step() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "$2"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel... "
    echo ""
}

# Function to execute command with output
run_command() {
    echo -e "${YELLOW}Executing:${NC} $1"
    echo ""
    eval "$1"
    local exit_code=$?
    echo ""
    return $exit_code
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        return 1
    fi
    return 0
}

# Header
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     GEPA Interactive Prompt Optimization Walkthrough         ║${NC}"
echo -e "${GREEN}║     Task: Banking77 Intent Classification                    ║${NC}"
echo -e "${GREEN}║     Model: gpt-oss-20b                                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# =============================================================================
# Step 0: Prerequisites Check
# =============================================================================
prompt_step "Step 0: Prerequisites Check" \
"Before we begin, let's verify all prerequisites are met:

${YELLOW}Required:${NC}
- Python 3.11+ with synth-ai package
- SYNTH_API_KEY environment variable (for backend authentication)
- OPENAI_API_KEY or GROQ_API_KEY (for LLM inference)

${YELLOW}Optional:${NC}
- cloudflared (for exposing local task app via tunnel)

We'll check each requirement now."

echo "Checking prerequisites..."
echo ""

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✓ Python installed: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

# Check synth-ai
if python3 -c "import synth_ai" 2>/dev/null; then
    echo -e "${GREEN}✓ synth-ai package installed${NC}"
else
    echo -e "${YELLOW}⚠ synth-ai package not found. Installing...${NC}"
    pip install synth-ai
fi

# Check SYNTH_API_KEY
if [ -n "$SYNTH_API_KEY" ]; then
    echo -e "${GREEN}✓ SYNTH_API_KEY is set: ${SYNTH_API_KEY:0:8}...${NC}"
else
    echo -e "${RED}✗ SYNTH_API_KEY not set${NC}"
    echo "  Set it with: export SYNTH_API_KEY=your-key"
    echo "  Get a key at: https://usesynth.ai/dashboard"
    exit 1
fi

# Check OPENAI_API_KEY or GROQ_API_KEY
if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "${GREEN}✓ OPENAI_API_KEY is set: ${OPENAI_API_KEY:0:8}...${NC}"
    LLM_PROVIDER="openai"
elif [ -n "$GROQ_API_KEY" ]; then
    echo -e "${GREEN}✓ GROQ_API_KEY is set: ${GROQ_API_KEY:0:8}...${NC}"
    LLM_PROVIDER="groq"
else
    echo -e "${RED}✗ Neither OPENAI_API_KEY nor GROQ_API_KEY is set${NC}"
    echo "  Set one with: export OPENAI_API_KEY=your-key"
    exit 1
fi

# Check cloudflared (optional)
if check_command cloudflared; then
    echo -e "${GREEN}✓ cloudflared installed (tunnel available)${NC}"
    HAS_CLOUDFLARED=true
else
    echo -e "${YELLOW}⚠ cloudflared not installed (will use local task app)${NC}"
    HAS_CLOUDFLARED=false
fi

echo ""
echo -e "${GREEN}All prerequisites met!${NC}"

# =============================================================================
# Step 1: Set Up Working Directory
# =============================================================================
prompt_step "Step 1: Set Up Working Directory" \
"We'll create a working directory to store:
- Generated API keys
- Configuration files
- Optimization results

Working directory: ${YELLOW}$WORKING_DIR${NC}"

run_command "mkdir -p $WORKING_DIR/results"
echo -e "${GREEN}✓ Working directory created${NC}"

# =============================================================================
# Step 2: Generate ENVIRONMENT_API_KEY
# =============================================================================
prompt_step "Step 2: Generate ENVIRONMENT_API_KEY" \
"The ENVIRONMENT_API_KEY authenticates requests between the optimizer
and your task app. This prevents unauthorized access to your task app.

${YELLOW}How it works:${NC}
1. Generate a unique API key
2. Register it with the Synth backend
3. Configure both task app and optimizer to use it

The key will be saved to: ${YELLOW}$WORKING_DIR/env_key.txt${NC}"

echo "Generating ENVIRONMENT_API_KEY..."

# Generate key using Python
ENV_KEY=$(python3 -c "
from synth_ai.learning.rl.secrets import mint_environment_api_key
print(mint_environment_api_key())
" 2>&1 | tail -1 | tr -d '\n' | tr -d '\r')

if [ -z "$ENV_KEY" ]; then
    echo -e "${YELLOW}⚠ Could not generate key via SDK, using UUID...${NC}"
    ENV_KEY=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
fi

echo "ENVIRONMENT_API_KEY=$ENV_KEY" > "$WORKING_DIR/env_key.txt"
export ENVIRONMENT_API_KEY="$ENV_KEY"

echo -e "${GREEN}✓ ENVIRONMENT_API_KEY generated: ${ENV_KEY:0:20}...${NC}"

# Register with backend
echo ""
echo "Registering key with backend..."
python3 -c "
import os
from pathlib import Path

try:
    from synth_ai.cli.lib.task_app_env import preflight_env_key
    env_file = Path('$WORKING_DIR/env_key.txt')
    preflight_env_key([env_file], crash_on_failure=False)
    print('✅ Key registered with backend')
except Exception as e:
    print(f'⚠️  Registration note: {e}')
    print('   Continuing anyway (key will work for local task apps)...')
" 2>&1

# =============================================================================
# Step 3: Start Task App
# =============================================================================
prompt_step "Step 3: Start Banking77 Task App" \
"The task app evaluates prompt candidates on the Banking77 dataset.

${YELLOW}Banking77 Dataset:${NC}
- 3,080 test examples
- 77 banking intent categories (e.g., 'card_arrival', 'transfer_fee_charged')
- Task: Classify customer queries into intent categories

${YELLOW}How the task app works:${NC}
1. Receives a prompt candidate from the optimizer
2. Tests it on samples from Banking77
3. Returns accuracy score (reward)

We'll start the task app on port 8001."

# Kill any existing task app
echo "Cleaning up existing processes on port 8001..."
lsof -ti :8001 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# Check if task app exists
TASK_APP_PATH="$COOKBOOK_DIR/task_app.py"
if [ ! -f "$TASK_APP_PATH" ]; then
    echo -e "${RED}Error: Task app not found at $TASK_APP_PATH${NC}"
    exit 1
fi

echo "Starting task app in background..."
echo "Task app path: $TASK_APP_PATH"

# Start task app with environment variables
ENVIRONMENT_API_KEY="$ENV_KEY" python3 "$TASK_APP_PATH" > "$WORKING_DIR/task_app.log" 2>&1 &
TASK_APP_PID=$!
echo "Task app PID: $TASK_APP_PID"

# Wait for task app to start
echo "Waiting for task app to start..."
sleep 5

# Verify task app is running
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Task app is running at http://localhost:8001${NC}"
    TASK_APP_URL="http://localhost:8001"
else
    echo -e "${RED}✗ Task app failed to start. Check $WORKING_DIR/task_app.log${NC}"
    cat "$WORKING_DIR/task_app.log"
    exit 1
fi

# Show task info
echo ""
echo "Fetching task info..."
curl -s -H "X-API-Key: $ENV_KEY" http://localhost:8001/task_info | python3 -m json.tool 2>/dev/null || echo "(Could not fetch task info)"

# =============================================================================
# Step 4: Create GEPA Configuration
# =============================================================================
prompt_step "Step 4: Create GEPA Configuration" \
"Now we'll create a GEPA configuration file optimized for this walkthrough.

${YELLOW}GEPA Parameters:${NC}
- population_size: 4 (prompts per generation)
- num_generations: 3 (evolution rounds)
- mutation_rate: 0.3 (30% chance of mutation)
- crossover_rate: 0.5 (50% chance of crossover)
- elite_count: 1 (top prompt preserved unchanged)

${YELLOW}Rollout Budget:${NC}
- budget: 100 (total prompt evaluations)
- Enough for meaningful optimization while keeping runtime short

Config will be saved to: ${YELLOW}$WORKING_DIR/gepa_config.toml${NC}"

# Create config file
cat > "$WORKING_DIR/gepa_config.toml" << 'EOF'
# GEPA Configuration for Banking77 Task
# Generated by run_interactive.sh walkthrough

[prompt_learning]
algorithm = "gepa"
task_app_url = "TASK_APP_URL_PLACEHOLDER"
results_folder = "RESULTS_FOLDER_PLACEHOLDER"

[prompt_learning.policy]
# LLM provider for generating responses
provider = "openai"
# Model for prompt evaluation (gpt-oss-20b equivalent)
model = "gpt-4o-mini"

[prompt_learning.termination_config]
# Maximum number of rollouts (prompt evaluations)
budget = 100
# Maximum time in seconds (30 minutes)
max_time_seconds = 1800
# Stop early if this score is reached
target_score = 0.95

[prompt_learning.gepa]
# Number of prompts per generation
population_size = 4
# Number of generations to evolve
num_generations = 3
# Probability of mutating each prompt component
mutation_rate = 0.3
# Probability of combining two prompts
crossover_rate = 0.5
# Number of top prompts to preserve unchanged
elite_count = 1

# Initial prompt template (starting point for evolution)
[prompt_learning.initial_prompt]
sections = [
    { role = "system", content = "You are a banking assistant that classifies customer intents. Given a customer's message, determine which of the 77 banking intent categories it belongs to. Categories include things like card_arrival, transfer_fee_charged, lost_or_stolen_card, etc. Respond with just the intent category name." }
]

# Display settings
[display]
show_curve = true
verbose_summary = true
show_trial_results = true
EOF

# Update placeholders with actual values
sed -i.bak "s|TASK_APP_URL_PLACEHOLDER|$TASK_APP_URL|" "$WORKING_DIR/gepa_config.toml"
sed -i.bak "s|RESULTS_FOLDER_PLACEHOLDER|$WORKING_DIR/results|" "$WORKING_DIR/gepa_config.toml"
rm -f "$WORKING_DIR/gepa_config.toml.bak"

echo -e "${GREEN}✓ Configuration created${NC}"
echo ""
echo "Configuration contents:"
echo "─────────────────────────"
cat "$WORKING_DIR/gepa_config.toml"
echo "─────────────────────────"

# =============================================================================
# Step 5: Submit GEPA Optimization Job
# =============================================================================
prompt_step "Step 5: Submit GEPA Optimization Job" \
"Now we'll submit the GEPA optimization job to the Synth backend.

${YELLOW}What happens during optimization:${NC}
1. Backend generates initial population of prompts
2. Each prompt is evaluated on Banking77 samples via your task app
3. Top prompts are selected, mutated, and crossed over
4. Process repeats for specified generations
5. Best prompt is returned

${YELLOW}Expected runtime:${NC}
- ~5-10 minutes depending on LLM response times
- You'll see progress updates as it runs

The --poll flag makes the CLI wait and show progress."

echo "Submitting GEPA optimization job..."
echo ""

# Run training with polling
run_command "uvx synth-ai train \"$WORKING_DIR/gepa_config.toml\" --poll --poll-timeout 3600 --stream-format cli"

TRAIN_EXIT_CODE=$?

# =============================================================================
# Step 6: Review Results
# =============================================================================
prompt_step "Step 6: Review Results" \
"Let's examine the optimization results.

${YELLOW}What to look for:${NC}
- Best score achieved
- Score progression across generations
- The optimized prompt text

Results are saved in: ${YELLOW}$WORKING_DIR/results/${NC}"

echo "Checking results directory..."
echo ""

if [ -d "$WORKING_DIR/results" ]; then
    echo "Results directory contents:"
    ls -la "$WORKING_DIR/results/" 2>/dev/null || echo "(empty)"
    echo ""

    # Look for result files
    for f in "$WORKING_DIR/results"/*.json; do
        if [ -f "$f" ]; then
            echo "Found result file: $f"
            echo "Contents (truncated):"
            echo "─────────────────────────"
            head -100 "$f" | python3 -m json.tool 2>/dev/null || cat "$f" | head -100
            echo "─────────────────────────"
        fi
    done
else
    echo "Results directory not found. Check if optimization completed successfully."
fi

# =============================================================================
# Cleanup
# =============================================================================
prompt_step "Step 7: Cleanup" \
"Let's clean up the background processes.

${YELLOW}We'll stop:${NC}
- Task app (PID: $TASK_APP_PID)

${YELLOW}Files preserved:${NC}
- Results: $WORKING_DIR/results/
- Config: $WORKING_DIR/gepa_config.toml
- Logs: $WORKING_DIR/task_app.log"

echo "Stopping task app..."
kill $TASK_APP_PID 2>/dev/null || true
echo -e "${GREEN}✓ Task app stopped${NC}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Walkthrough Complete!                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}What you learned:${NC}"
echo "  1. How to set up ENVIRONMENT_API_KEY for task app authentication"
echo "  2. How to configure GEPA for prompt optimization"
echo "  3. How GEPA evolves prompts using genetic algorithms"
echo "  4. How to submit jobs and monitor progress"
echo ""
echo -e "${YELLOW}Files created:${NC}"
echo "  - $WORKING_DIR/env_key.txt (API key)"
echo "  - $WORKING_DIR/gepa_config.toml (configuration)"
echo "  - $WORKING_DIR/results/ (optimization results)"
echo "  - $WORKING_DIR/task_app.log (task app logs)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Try MIPRO: bash ../mipro/run_interactive.sh"
echo "  2. Compare results between GEPA and MIPRO"
echo "  3. Use the optimized prompt in production"
echo "  4. Experiment with different GEPA parameters"
echo ""
echo -e "${YELLOW}Using the optimized prompt:${NC}"
echo "  python3 << 'PYTHON'"
echo "  import json"
echo "  from openai import OpenAI"
echo ""
echo "  # Load optimized prompt from results"
echo "  with open('$WORKING_DIR/results/best_prompt.json') as f:"
echo "      prompt = json.load(f)"
echo ""
echo "  client = OpenAI()"
echo "  response = client.chat.completions.create("
echo "      model='gpt-4o-mini',"
echo "      messages=["
echo "          {'role': 'system', 'content': prompt['content']},"
echo "          {'role': 'user', 'content': 'I want to cancel my card'},"
echo "      ]"
echo "  )"
echo "  print(response.choices[0].message.content)"
echo "  PYTHON"
echo ""
