# Supervised Fine-Tuning (SFT) Cookbook (Official)

Complete guide to supervised fine-tuning for Banking77 intent classification using both SDK and CLI approaches.

## What you'll build
- A fine-tuned language model for banking intent classification (Banking77 dataset from HuggingFace)
- Training pipeline from dataset → training → inference
- Evaluation workflow to measure improvement over base model

## Prerequisites
- Python 3.11+
- `synth-ai` package: `pip install synth-ai`
- API keys: `SYNTH_API_KEY`
- HuggingFace `datasets` library (auto-installed with synth-ai)

## Dataset

This cookbook uses the **Banking77** dataset from HuggingFace:
- **Repository**: `banking77` (HuggingFace)
- **URL**: https://huggingface.co/datasets/banking77
- **Train examples**: 10,003
- **Test examples**: 3,080
- **Classes**: 77 banking intent categories

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
| Custom data preprocessing | SDK | Programmatic dataset handling |
| CI/CD pipelines | CLI | Shell-scriptable |
| Progress monitoring | SDK | Custom callbacks |
| Using fine-tuned model | SDK | Direct inference integration |

## Project Structure

```
sft/
├── README.md                 # This file
├── sdk/
│   ├── basic.py              # Basic SDK example (downloads Banking77)
│   ├── advanced.py           # Advanced features (checkpoints, eval)
│   └── inference.py          # Using fine-tuned model
├── cli/
│   ├── run.sh                # Training via CLI
│   └── .env.example          # Environment template
└── configs/
    └── sft.toml              # Training configuration
```

Note: Training data is downloaded from HuggingFace at runtime, not stored in the repo.

## Step-by-Step Guide

### 1. Prepare Dataset (Optional - Auto-generated)

The SDK/CLI can automatically download Banking77 from HuggingFace. For custom datasets, use JSONL format:

```jsonl
{"messages": [{"role": "system", "content": "You are a banking assistant."}, {"role": "user", "content": "I want to cancel my card"}, {"role": "assistant", "content": "card_about_to_expire"}]}
```

### 2. Configure Training

Edit `configs/sft.toml`:

```toml
[algorithm]
type = "offline"
variety = "fft"
method = "sft"

[job]
model = "Qwen/Qwen3-0.6B"
data = "../data/train.jsonl"

[hyperparameters]
n_epochs = 3
learning_rate = 1e-5
batch_size = 4
```

### 3. Run Training

**SDK approach:**
```python
from synth_ai.sdk.api.train.sft import SFTJob

job = SFTJob.from_config("configs/sft.toml")
job_id = job.submit()
result = job.poll_until_complete()
model_id = job.get_fine_tuned_model()
print(f"Fine-tuned model: {model_id}")
```

**CLI approach:**
```bash
uvx synth-ai train configs/sft.toml --poll
```

### 4. Use Fine-Tuned Model

```python
from synth_ai.sdk.inference import InferenceClient

client = InferenceClient(api_key=os.environ["SYNTH_API_KEY"])
response = await client.create_chat_completion(
    model="ft:Qwen/Qwen3-0.6B:banking77:abc123",
    messages=[
        {"role": "system", "content": "You are a banking assistant."},
        {"role": "user", "content": "How do I transfer money?"}
    ]
)
print(response["choices"][0]["message"]["content"])  # "transfer"
```

## Configuration Reference

### SFT Config (`configs/sft.toml`)

```toml
[algorithm]
type = "offline"
variety = "fft"  # or "lora"
method = "sft"

[job]
model = "Qwen/Qwen3-0.6B"
data = "../data/train.jsonl"
validation_data = "../data/validation.jsonl"
suffix = "banking77-intent"

[compute]
gpu_type = "H100"
gpu_count = 1

[hyperparameters]
n_epochs = 3
learning_rate = 1e-5
batch_size = 4
warmup_ratio = 0.1
weight_decay = 0.01
max_seq_length = 512
```

## Banking77 Intent Categories

The model learns to classify queries into 77 categories including:
- `activate_my_card`, `card_arrival`, `card_about_to_expire`
- `transfer`, `transfer_fee_charged`, `transfer_timing`
- `balance_not_updated_after_bank_transfer`, `check_balance`
- `lost_or_stolen_card`, `declined_card_payment`
- ... and 67 more

## Integration Tests

The SDK and CLI functionality demonstrated in this cookbook is covered by integration tests in `synth-ai/tests/integration/`:

- `training/test_sft_sdk.py` - Tests for FtClient, dataset validation, and job creation
- `training/test_cli_train.py` - Tests for CLI command parsing and execution
- `inference/test_inference_client.py` - Tests for InferenceClient used with fine-tuned models

Run the tests:
```bash
cd synth-ai
pytest tests/integration/training/test_sft_sdk.py -v
pytest tests/integration/inference/test_inference_client.py -v
```

## See Also

- [Banking77 Task App](../../dev/task_apps/banking77/) - Full task app implementation
- [Prompt Learning Cookbook](../prompt_learning/) - Optimize prompts instead of weights
- [RL Cookbook](../rl/) - Reinforcement learning from feedback
- [SDK Documentation](https://docs.usesynth.ai/sdk/training/sft)
- [CLI Reference](https://docs.usesynth.ai/cli/train)
