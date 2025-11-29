# Reinforcement Learning (RL) Cookbook (Official)

Complete guide to reinforcement learning from feedback for heart disease classification using both SDK and CLI approaches.

## What you'll build
- An RL-trained model for heart disease prediction (buio/heart-disease dataset from HuggingFace)
- A Task App that evaluates model predictions and provides rewards
- Training pipeline from config → rollouts → optimized model

## Prerequisites
- Python 3.11+
- `synth-ai` package: `pip install synth-ai`
- API keys: `SYNTH_API_KEY`, `ENVIRONMENT_API_KEY`, `OPENAI_API_KEY` (or compatible)
- HuggingFace `datasets` library (auto-installed with synth-ai)
- Optional: `cloudflared` for exposing local task apps

## Dataset

This cookbook uses the **Heart Disease** dataset from HuggingFace:
- **Repository**: `buio/heart-disease` (HuggingFace)
- **URL**: https://huggingface.co/datasets/buio/heart-disease
- **Train examples**: 303 patient records
- **Features**: 14 medical metrics (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- **Target**: Binary classification (0 = no disease, 1 = heart disease)

The dataset is automatically downloaded and cached by the `datasets` library.

## Quick Start

### Option A: SDK (Pure Python)

```bash
cd sdk
python basic.py
```

### Option B: CLI (Command Line)

```bash
cd cli
bash run.sh
```

## SDK vs CLI: When to Use Each

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Quick training job | CLI | Single command, no code |
| Integration into Python scripts | SDK | Programmatic control |
| Custom reward functions | SDK | Modify task app in code |
| In-process task apps | SDK | `InProcessTaskApp` context manager |
| CI/CD pipelines | CLI | Shell-scriptable |

## Understanding RL Training

RL training optimizes a model through interaction with an environment (your Task App):

```
┌─────────────────┐         ┌──────────────────┐
│  RL Optimizer   │  HTTP   │  Your Task App   │
│  (GRPO/PPO)     │ ──────> │  (Heart Disease) │
│                 │         │                  │
│  Generates      │         │  Evaluates       │
│  responses      │ <────── │  predictions,    │
│                 │  reward │  returns reward  │
└─────────────────┘         └──────────────────┘
```

### Key Concepts

- **Rollout**: One episode of model generating a response and receiving reward
- **Reward**: Signal (0.0 or 1.0) indicating correctness of prediction
- **Policy**: The model being trained
- **GRPO**: Group Relative Policy Optimization (recommended for LLMs)

## Project Structure

```
rl/
├── README.md                 # This file
├── sdk/
│   ├── basic.py              # Basic SDK example
│   ├── advanced.py           # Advanced features (metrics, checkpoints)
│   └── in_process.py         # In-process task app example
├── cli/
│   ├── run.sh                # Training via CLI
│   └── .env.example          # Environment template
├── configs/
│   └── rl.toml               # Training configuration
└── task_app.py               # Heart disease evaluation task app
```

## Step-by-Step Guide

### 1. Set Up Environment

```bash
# Create .env file
cp cli/.env.example .env

# Edit with your keys
vim .env  # Add SYNTH_API_KEY, ENVIRONMENT_API_KEY, OPENAI_API_KEY
```

### 2. Understand the Task

The heart disease task app evaluates model predictions on patient data:

- **Input**: Patient features (age, cholesterol, blood pressure, etc.)
- **Output**: Classification (1 = heart disease, 0 = no disease)
- **Reward**: 1.0 if prediction matches ground truth, 0.0 otherwise

### 3. Start the Task App

```bash
# Local development (uses existing task app from dev/task_apps)
python -m synth_ai.sdk.task.server --app heartdisease --port 8114

# Or use the cookbook's simplified task app
python task_app.py

# Or expose via tunnel for remote training
cloudflared tunnel --url http://localhost:8114
```

### 4. Run Training

**SDK approach:**
```python
from synth_ai.sdk.api.train.rl import RLJob

job = RLJob.from_config("configs/rl.toml")
job_id = job.submit()
result = job.poll_until_complete()
print(f"Final reward: {result.get('final_reward')}")
```

**CLI approach:**
```bash
uvx synth-ai train configs/rl.toml --poll
```

### 5. Evaluate Results

```python
# Get training metrics
metrics = job.get_metrics()
print(f"Steps: {len(metrics['steps'])}")
print(f"Final reward: {metrics['steps'][-1]['reward']}")
```

## Configuration Reference

### RL Config (`configs/rl.toml`)

```toml
[algorithm]
type = "online"
variety = "grpo"

[policy]
model = "Qwen/Qwen3-0.6B"
provider = "synth"

[hyperparameters]
n_rollouts = 100
batch_size = 4
learning_rate = 1e-5
kl_coef = 0.1
```

## Heart Disease Features

The model receives patient data with these features:
- `age`: Patient age in years
- `sex`: 0 = Female, 1 = Male
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (0/1)
- `restecg`: Resting ECG results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (0/1)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of ST segment (0-2)
- `ca`: Number of major vessels colored by fluoroscopy (0-3)
- `thal`: Thalassemia type (fixed, normal, reversible)

## Task App Implementation

The task app uses the `heart_disease_classify` tool for structured predictions:

```python
tool_spec = {
    "type": "function",
    "function": {
        "name": "heart_disease_classify",
        "description": "Submit your classification prediction.",
        "parameters": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "description": "'1' for heart disease, '0' for no disease",
                    "enum": ["0", "1"],
                }
            },
            "required": ["classification"],
        },
    },
}
```

## Common Patterns

### Using the Trained Model

```python
from synth_ai.sdk.inference import InferenceClient

client = InferenceClient(api_key=os.environ["SYNTH_API_KEY"])

patient_data = """
age: 55
sex: 1
cp: 2
trestbps: 140
chol: 250
fbs: 0
restecg: 1
thalach: 145
exang: 1
oldpeak: 2.3
slope: 2
ca: 1
thal: normal
"""

response = await client.create_chat_completion(
    model=trained_model_id,
    messages=[
        {"role": "system", "content": "Classify: 1 for heart disease, 0 for no disease."},
        {"role": "user", "content": f"Patient Features:\n{patient_data}"},
    ]
)
```

### Monitoring Training Progress

```python
def on_metrics(metrics: dict) -> None:
    step = metrics.get("step", 0)
    reward = metrics.get("reward", 0)
    loss = metrics.get("loss", 0)
    print(f"Step {step}: reward={reward:.4f}, loss={loss:.4f}")

result = job.poll_until_complete(on_status=on_metrics)
```

## Troubleshooting

### Task App Not Reachable

1. Check task app is running: `curl http://localhost:8114/health`
2. If using tunnel, verify HTTPS URL is correct
3. Ensure `ENVIRONMENT_API_KEY` matches in both task app and config

### Low Rewards

1. Review the model's predictions in logs
2. Try increasing `n_rollouts` for more training
3. Consider using a larger base model

### Training Unstable

1. Reduce `learning_rate`
2. Increase `kl_coef` to constrain updates
3. Use smaller `batch_size`

## Integration Tests

The SDK and CLI functionality demonstrated in this cookbook is covered by integration tests in `synth-ai/tests/integration/`:

- `training/test_rl_sdk.py` - Tests for RLClient, config validation, job creation, and metrics
- `training/test_cli_train.py` - Tests for CLI command parsing and execution

Run the tests:
```bash
cd synth-ai
pytest tests/integration/training/test_rl_sdk.py -v
pytest tests/integration/training/test_cli_train.py -v
```

## See Also

- [Heart Disease Task App](../../dev/task_apps/other_langprobe_benchmarks/heartdisease_task_app.py) - Full task app implementation
- [Prompt Learning Cookbook](../prompt_learning/) - Optimize prompts instead of weights
- [SFT Cookbook](../sft/) - Pre-train with supervised data first
- [SDK Documentation](https://docs.usesynth.ai/sdk/training/rl)
- [CLI Reference](https://docs.usesynth.ai/cli/train)
