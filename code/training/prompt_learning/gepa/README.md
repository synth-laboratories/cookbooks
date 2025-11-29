# GEPA: Genetic Evolution of Prompt Architectures

Interactive walkthrough for prompt optimization using GEPA on the Banking77 intent classification task.

## Overview

GEPA uses **genetic algorithms** to evolve prompt structures. It treats prompts as "chromosomes" that can be:
- **Mutated**: Random modifications to prompt sections
- **Crossed over**: Combining parts from two successful prompts
- **Selected**: Keeping the best-performing variants

This approach is particularly effective for:
- Complex prompt structures with multiple sections
- Multi-component prompts (system + few-shot + chain-of-thought)
- Exploring diverse prompt variations

## Quick Start

```bash
# Run the interactive walkthrough
bash run_interactive.sh
```

The script guides you through each step with explanations.

## Prerequisites

| Requirement | Description |
|-------------|-------------|
| Python 3.11+ | With synth-ai package |
| SYNTH_API_KEY | Backend authentication ([get one here](https://usesynth.ai)) |
| OPENAI_API_KEY | LLM inference (or GROQ_API_KEY) |

## What This Walkthrough Does

```
┌─────────────────────────────────────────────────────────────────┐
│                    GEPA Optimization Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Prerequisites Check                                     │
│     └─→ Verify Python, API keys, synth-ai package               │
│                                                                  │
│  Step 2: Generate ENVIRONMENT_API_KEY                            │
│     └─→ Create authentication key for task app                  │
│                                                                  │
│  Step 3: Start Banking77 Task App                                │
│     └─→ Local server evaluating prompts on Banking77            │
│                                                                  │
│  Step 4: Create GEPA Configuration                               │
│     └─→ Set population_size, generations, mutation rate         │
│                                                                  │
│  Step 5: Submit Optimization Job                                 │
│     └─→ Backend evolves prompts, task app evaluates             │
│                                                                  │
│  Step 6: Review Results                                          │
│     └─→ Best prompt, score progression, artifacts               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Guide

### Step 1: Environment Setup

The script checks for:
```bash
# Required environment variables
export SYNTH_API_KEY="your-synth-api-key"
export OPENAI_API_KEY="your-openai-key"  # or GROQ_API_KEY
```

**Expected output:**
```
Checking prerequisites...

✓ Python installed: Python 3.11.4
✓ synth-ai package installed
✓ SYNTH_API_KEY is set: sk-synth-...
✓ OPENAI_API_KEY is set: sk-...
✓ cloudflared installed (tunnel available)

All prerequisites met!
```

### Step 2: Generate ENVIRONMENT_API_KEY

This creates an authentication key for the task app:

```bash
Generating ENVIRONMENT_API_KEY...
✓ ENVIRONMENT_API_KEY generated: 8f3a2b1c-4d5e-6f7g...

Registering key with backend...
✅ Key registered with backend
```

### Step 3: Start Task App

The Banking77 task app starts locally:

```bash
Starting task app in background...
Task app PID: 12345
Waiting for task app to start...
✓ Task app is running at http://localhost:8001

Fetching task info...
{
    "task_app_id": "banking77_intent_classification",
    "dataset_size": 3080,
    "labels": ["activate_my_card", "age_limit", "apple_pay_or_google_pay", ...]
}
```

### Step 4: Create Configuration

The script generates `gepa_config.toml`:

```toml
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.gepa]
population_size = 4      # Prompts per generation
num_generations = 3      # Evolution rounds
mutation_rate = 0.3      # 30% mutation chance
crossover_rate = 0.5     # 50% crossover chance
elite_count = 1          # Top prompt preserved
```

### Step 5: Submit Optimization Job

```bash
Submitting GEPA optimization job...

Executing: uvx synth-ai train "/tmp/gepa_walkthrough/gepa_config.toml" --poll

========================================
GEPA Prompt Optimization
========================================
Job ID: gepa_banking77_1234567890
Status: running

Generation 1/3:
  Evaluating prompt 1/4... score: 0.42
  Evaluating prompt 2/4... score: 0.48
  Evaluating prompt 3/4... score: 0.45
  Evaluating prompt 4/4... score: 0.51
  Best this generation: 0.51

Generation 2/3:
  Evaluating prompt 1/4... score: 0.53
  Evaluating prompt 2/4... score: 0.56
  Evaluating prompt 3/4... score: 0.52
  Evaluating prompt 4/4... score: 0.58
  Best this generation: 0.58

Generation 3/3:
  Evaluating prompt 1/4... score: 0.59
  Evaluating prompt 2/4... score: 0.62
  Evaluating prompt 3/4... score: 0.60
  Evaluating prompt 4/4... score: 0.64
  Best this generation: 0.64

========================================
Optimization Complete
========================================
Best Score: 0.64
Generations: 3
Total Evaluations: 100
```

### Step 6: Review Results

```bash
Results directory contents:
-rw-r--r--  1 user  staff  2048 Nov 28 12:00 best_prompt.json
-rw-r--r--  1 user  staff  4096 Nov 28 12:00 score_history.json
-rw-r--r--  1 user  staff  1024 Nov 28 12:00 job_metadata.json

Found result file: /tmp/gepa_walkthrough/results/best_prompt.json
Contents:
{
  "role": "system",
  "content": "You are an expert banking intent classifier. Analyze the customer's message and identify the specific banking intent from these 77 categories..."
}
```

## Configuration Reference

### GEPA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 4 | Number of prompts per generation |
| `num_generations` | 3 | Number of evolution rounds |
| `mutation_rate` | 0.3 | Probability of mutating each component |
| `crossover_rate` | 0.5 | Probability of combining two prompts |
| `elite_count` | 1 | Top prompts preserved unchanged |

### Termination Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget` | 100 | Maximum prompt evaluations |
| `max_time_seconds` | 1800 | Maximum runtime (30 min) |
| `target_score` | 0.95 | Early stopping threshold |

### Tuning Tips

**For faster iteration (walkthroughs):**
```toml
population_size = 4
num_generations = 3
budget = 100
```

**For better results (production):**
```toml
population_size = 10
num_generations = 10
budget = 1000
```

## How GEPA Works

```
Generation 0 (Initial Population)
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Prompt 1│ │Prompt 2│ │Prompt 3│ │Prompt 4│
│(base)  │ │(mutant)│ │(mutant)│ │(mutant)│
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    │          │          │          │
    ▼          ▼          ▼          ▼
┌────────────────────────────────────────┐
│           EVALUATION                    │
│  Test each prompt on Banking77 samples │
└────────────────────────────────────────┘
    │          │          │          │
    ▼          ▼          ▼          ▼
 0.42       0.48       0.45       0.51  (scores)
    │          │          │          │
    └──────────┴──────────┴──────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│           SELECTION                     │
│  Keep top performers (elites)          │
│  Select parents for next generation    │
└────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│    CROSSOVER    │ │    MUTATION     │
│  Combine parts  │ │  Random changes │
│  from 2 parents │ │  to components  │
└─────────────────┘ └─────────────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
Generation 1 (New Population)
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Elite   │ │Child 1 │ │Child 2 │ │Child 3 │
│(0.51)  │ │(cross) │ │(mutant)│ │(cross) │
└────────┘ └────────┘ └────────┘ └────────┘

... repeat for num_generations ...
```

## Troubleshooting

### Task App Not Starting

```bash
# Check if port is in use
lsof -i :8001

# Kill existing process
lsof -ti :8001 | xargs kill -9

# Check task app logs
cat /tmp/gepa_walkthrough/task_app.log
```

### Low Optimization Scores

1. **Increase budget**: More evaluations = more exploration
   ```toml
   budget = 500
   ```

2. **Increase population**: More diversity
   ```toml
   population_size = 8
   ```

3. **Adjust mutation rate**: Higher = more exploration
   ```toml
   mutation_rate = 0.4
   ```

### Job Timeout

Increase the time limit:
```toml
max_time_seconds = 3600  # 1 hour
```

## Comparison with MIPRO

| Aspect | GEPA | MIPRO |
|--------|------|-------|
| **Approach** | Genetic evolution | Bayesian optimization |
| **Best for** | Complex structures | Instruction tuning |
| **Exploration** | High (mutations) | Moderate (TPE) |
| **Convergence** | Slower | Faster |

Try both on the same task:
```bash
# Run GEPA
bash run_interactive.sh

# Run MIPRO (same task, same model)
bash ../mipro/run_interactive.sh
```

## Files Created

After running the walkthrough:

```
/tmp/gepa_walkthrough/
├── env_key.txt           # ENVIRONMENT_API_KEY
├── gepa_config.toml      # Generated configuration
├── task_app.log          # Task app output
└── results/
    ├── best_prompt.json  # Optimized prompt
    ├── score_history.json # Score progression
    └── job_metadata.json  # Job details
```

## Next Steps

1. **Try MIPRO**: `bash ../mipro/run_interactive.sh`
2. **Compare results**: Check score and prompt quality
3. **Production use**: Copy optimized prompt to your application
4. **Experiment**: Adjust parameters and re-run

## See Also

- [MIPRO Walkthrough](../mipro/) - Alternative optimization algorithm
- [Main README](../README.md) - Overview of all prompt learning approaches
- [SDK Examples](../sdk/) - Programmatic Python interface
- [CLI Scripts](../cli/) - Non-interactive command-line runners
