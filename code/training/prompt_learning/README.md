# Prompt Learning Cookbook

Complete guide to prompt optimization using MIPRO and GEPA algorithms, with interactive walkthroughs, SDK examples, and CLI approaches.

## What you'll build
- Optimized prompts using MIPRO (instruction/demo optimization) or GEPA (genetic evolution)
- A Task App that evaluates prompt candidates
- Automated pipeline from config → training → results

## Quick Start: Interactive Walkthroughs

The fastest way to learn prompt optimization is through our **interactive walkthroughs**. Both use the Banking77 intent classification task with gpt-oss-20b for easy comparison.

```bash
# GEPA: Genetic evolution of prompts
cd gepa && bash run_interactive.sh

# MIPRO: Bayesian optimization with TPE
cd mipro && bash run_interactive.sh
```

These scripts guide you through each step with explanations, showing actual commands and expected outputs.

## Algorithm Comparison

| Aspect | GEPA | MIPRO |
|--------|------|-------|
| **Approach** | Genetic evolution | Bayesian optimization (TPE) |
| **Instruction generation** | Mutation operators | Meta-model generation |
| **Demo selection** | Evolved with prompt | Jointly optimized |
| **Convergence** | Slower, thorough | Faster, efficient |
| **Exploration** | High (diverse mutations) | Moderate (guided search) |
| **Best for** | Complex prompt structures | Instruction-following tasks |

### Choose GEPA when:
- You need to optimize complex prompt structures (multiple sections)
- You want diverse exploration of prompt space
- Task benefits from creative prompt mutations
- No clear demo examples available

### Choose MIPRO when:
- Task has clear instruction-following structure
- Few-shot examples improve performance
- You want faster convergence
- Good training examples available for demonstrations

## Prerequisites
- Python 3.11+
- `synth-ai` package: `pip install synth-ai`
- API keys: `SYNTH_API_KEY`, `ENVIRONMENT_API_KEY`, `OPENAI_API_KEY` (or compatible)
- Optional: `cloudflared` for exposing local task apps

## Project Structure

```
prompt_learning/
├── README.md                 # This file
├── gepa/                     # GEPA interactive walkthrough
│   ├── README.md             # GEPA-specific guide with expected outputs
│   ├── run_interactive.sh    # Interactive end-to-end script
│   ├── config.toml           # GEPA configuration (Banking77 + gpt-oss-20b)
│   └── expected_outputs/     # Example command outputs
├── mipro/                    # MIPRO interactive walkthrough
│   ├── README.md             # MIPRO-specific guide with expected outputs
│   ├── run_interactive.sh    # Interactive end-to-end script
│   ├── config.toml           # MIPRO configuration (Banking77 + gpt-oss-20b)
│   └── expected_outputs/     # Example command outputs
├── sdk/                      # SDK examples (pure Python)
│   ├── basic.py              # Basic SDK example
│   ├── advanced.py           # Advanced features
│   └── in_process.py         # In-process task app example
├── cli/                      # CLI runners (non-interactive)
│   ├── run_gepa.sh           # Simple GEPA runner
│   ├── run_mipro.sh          # Simple MIPRO runner
│   └── .env.example          # Environment template
├── configs/                  # Base configurations
│   ├── gepa.toml             # GEPA base config
│   └── mipro.toml            # MIPRO base config
└── task_app.py               # Banking77 task app
```

## Approaches

### Option A: Interactive Walkthroughs (Recommended for Learning)

Best for understanding how the algorithms work:

```bash
# GEPA walkthrough
cd gepa && bash run_interactive.sh

# MIPRO walkthrough
cd mipro && bash run_interactive.sh
```

Features:
- Step-by-step guidance with explanations
- User approval before each step
- Educational context for each operation
- Actual command outputs shown

### Option B: SDK (Pure Python)

Best for integration into Python applications:

```bash
cd sdk
python basic.py
```

```python
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

job = PromptLearningJob.from_config("configs/gepa.toml")
job_id = job.submit()
result = job.poll_until_complete()
print(f"Best score: {result['best_score']}")
```

### Option C: CLI (Command Line)

Best for scripts and CI/CD:

```bash
cd cli
bash run_gepa.sh   # or run_mipro.sh
```

Or directly:
```bash
uvx synth-ai train configs/gepa.toml --poll
```

## SDK vs CLI: When to Use Each

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Learning the algorithms | Interactive | Step-by-step explanations |
| Quick one-off optimization | CLI | Single command, no code |
| Integration into Python scripts | SDK | Programmatic control |
| CI/CD pipelines | CLI | Shell-scriptable |
| Custom polling/error handling | SDK | Full control over flow |
| In-process task apps | SDK | `InProcessTaskApp` context manager |

## Dataset

This cookbook uses the **Banking77** dataset from HuggingFace:
- **Repository**: `banking77` (HuggingFace)
- **URL**: https://huggingface.co/datasets/banking77
- **Test examples**: 3,080 (used for prompt evaluation)
- **Classes**: 77 banking intent categories

The dataset is automatically downloaded when the task app starts.

## Step-by-Step Guide

### 1. Set Up Environment

```bash
# Create .env file
cp cli/.env.example .env

# Edit with your keys
vim .env  # Add SYNTH_API_KEY, ENVIRONMENT_API_KEY, OPENAI_API_KEY
```

### 2. Start the Task App

```bash
# Local development
python task_app.py

# Or expose via tunnel
cloudflared tunnel --url http://localhost:8001
```

### 3. Run Optimization

**Interactive approach (recommended for learning):**
```bash
cd gepa && bash run_interactive.sh  # or cd mipro
```

**SDK approach:**
```python
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

job = PromptLearningJob.from_config("configs/gepa.toml")
job_id = job.submit()
result = job.poll_until_complete()
print(f"Best score: {result['best_score']}")
```

**CLI approach:**
```bash
uvx synth-ai train configs/gepa.toml --poll
```

### 4. Extract Results

```python
# Get optimized prompt
results = job.get_results()
best_prompt = results['best_prompt']

# Save for production use
import json
with open("optimized_prompt.json", "w") as f:
    json.dump(best_prompt, f, indent=2)
```

## Configuration Reference

### GEPA Config (`configs/gepa.toml`)

```toml
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.policy]
provider = "openai"
model = "gpt-4o-mini"

[prompt_learning.termination_config]
budget = 100
max_time_seconds = 1800

[prompt_learning.gepa]
population_size = 4      # Prompts per generation
num_generations = 3      # Evolution rounds
mutation_rate = 0.3      # 30% mutation chance
crossover_rate = 0.5     # 50% crossover chance
elite_count = 1          # Top prompt preserved
```

See [gepa/README.md](gepa/README.md) for detailed parameter explanations.

### MIPRO Config (`configs/mipro.toml`)

```toml
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.policy]
provider = "openai"
model = "gpt-4o-mini"

[prompt_learning.mipro]
num_iterations = 10           # TPE optimization rounds
top_k = 2                     # Candidates to keep
num_bootstrap_seeds = 3       # Bootstrap samples
num_instructions_per_module = 4  # Instruction variants
num_demos = 2                 # Few-shot examples
optimize_instructions = true
optimize_demos = true
```

See [mipro/README.md](mipro/README.md) for detailed parameter explanations.

## Common Patterns

### Using Optimized Prompts in Production

```python
import json
from openai import OpenAI

# Load optimized prompt
with open("optimized_prompt.json") as f:
    prompt = json.load(f)

# Use in production
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt["sections"][0]["content"]},
        {"role": "user", "content": user_input},
    ]
)
```

### Comparing GEPA vs MIPRO

Run both algorithms on the same task:

```bash
# Run GEPA
cd gepa && bash run_interactive.sh

# Run MIPRO
cd mipro && bash run_interactive.sh

# Compare results
cat /tmp/gepa_walkthrough/results/best_prompt.json
cat /tmp/mipro_walkthrough/results/best_prompt.json
```

### Comparing Multiple Runs (SDK)

```python
import pandas as pd
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

runs = []
for config in ["configs/gepa.toml", "configs/mipro.toml"]:
    job = PromptLearningJob.from_config(config)
    job.submit()
    result = job.poll_until_complete()
    runs.append({
        "algorithm": config.split("/")[-1].split(".")[0],
        "best_score": result["best_score"],
        "job_id": job.job_id,
    })

df = pd.DataFrame(runs)
print(df.sort_values("best_score", ascending=False))
```

## Troubleshooting

### Task App Not Reachable

1. Check task app is running: `curl http://localhost:8001/health`
2. If using tunnel, verify HTTPS URL is correct
3. Ensure `ENVIRONMENT_API_KEY` matches in both task app and config

### Optimization Stalls

1. Check task app logs for errors
2. Increase `max_time_seconds` in config
3. Try different `model` (e.g., `gpt-4o` for complex tasks)

### Low Scores

1. Review your reward function in `task_app.py`
2. Ensure dataset has clear correct answers
3. Try increasing `budget` for more exploration
4. Adjust algorithm-specific parameters (see algorithm READMEs)

## Integration Tests

The SDK and CLI functionality demonstrated in this cookbook is covered by integration tests in `synth-ai/tests/integration/`:

- `training/test_prompt_learning_sdk.py` - Tests for PromptLearningJob configuration, submission, and polling
- `training/test_cli_train.py` - Tests for CLI command parsing and execution

Run the tests:
```bash
cd synth-ai
pytest tests/integration/training/test_prompt_learning_sdk.py -v
pytest tests/integration/training/test_cli_train.py -v
```

## See Also

- [GEPA Interactive Walkthrough](gepa/) - Genetic evolution with step-by-step guidance
- [MIPRO Interactive Walkthrough](mipro/) - Bayesian optimization with TPE
- [Polyglot Task Apps](../polyglot/) - Multi-language task app examples
- [SFT Cookbook](../sft/) - Supervised fine-tuning guide
- [RL Cookbook](../rl/) - Reinforcement learning guide
- [SDK Documentation](https://docs.usesynth.ai/sdk/training/prompt-optimization)
- [CLI Reference](https://docs.usesynth.ai/cli/train)
