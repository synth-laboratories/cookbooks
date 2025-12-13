# Crafter Verifier Graph Calibration

Calibrate a verifier graph that judges Crafter game agent performance.

## What You'll Build

A **verifier graph** that:
1. Receives a game execution trace (actions, observations, rewards)
2. Analyzes multiple performance aspects (survival, exploration, achievements)
3. Produces a final score (0-1) + reasoning

```
[trace] → [digest_aspects] → [combine_scores] → {score, reasoning}
```

## Quick Start

```bash
# Set API keys
export SYNTH_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# Run calibration
python run_crafter_verifier.py
```

## Expected Output

```
🎮 Crafter Verifier Graph Calibration
============================================================
   Traces: 16 (train), 4 (val)
   Graph type: verifier
   Structure: dag
   Generations: 3

📊 Streaming optimization progress...
   [7a1beebb] score=0.908
[Gen 1] best=0.908, archive=1
   [812eb5d3] score=0.625
   [58b726ef] score=0.540
[Gen 2] best=0.908, archive=2
   [c9e712a3] score=0.999
[Gen 3] best=0.999, archive=2

✅ Calibration successful!
   Best score: 0.999
   Generations: 3

📊 Best Graph (prompts + UML):
--------------------------------------------------
Prompts (per node):

--- digest_aspects ---
You are an expert judge of Crafter execution traces.
Analyze the agent's performance across 4 criteria:
- survival (health, hunger management)
- exploration (new areas, resources)
- achievement_progress (normalized 0-22)
- efficiency (purposeful vs random actions)

Dataflow (UML):
@startuml
start
digest_aspects --> combine_scores
combine_scores --> end
@enduml
--------------------------------------------------
```

## How It Works

1. **Dataset Upload**: Traces with gold scores are sent to backend
2. **Initial Generation**: LLM creates a verifier graph
3. **Calibration**: Graph outputs scored against gold labels (MAE + LLM judge)
4. **Evolution**: Proposer improves scoring accuracy
5. **Output**: Calibrated verifier (prompts + dataflow)

## Scoring

Verifier graphs are scored on **agreement** with gold labels:

- **MAE-based**: `reward = 1.0 - abs(pred - gold)`
- **LLM Judge**: Rubric-based evaluation considering:
  - Score accuracy vs gold
  - Reasoning quality
  - Aspect coverage

## Configuration

See `config.toml`:

```toml
[graph_optimization]
dataset_name = "crafter_verifier"
graph_type = "verifier"
graph_structure = "dag"

[graph_optimization.proposer]
model = "gpt-4.1"

[graph_optimization.population]
num_generations = 3
children_per_generation = 2
```

## Dataset Format

The calibration requires traces with gold scores:

```json
{
  "name": "crafter_verifier_calibration",
  "task_description": "Evaluate Crafter agent execution traces",
  "examples": [
    {
      "id": "trace_001",
      "input": {
        "trace": { "events": [...], "observations": [...] },
        "trace_id": "session_abc123"
      },
      "expected": {
        "score": 0.18,
        "achievements_count": 4
      }
    }
  ]
}
```

## Files

```
crafter/
├── README.md              # This file
├── config.toml            # GEPA configuration
├── dataset.json           # Sample traces (optional)
├── run_crafter_verifier.py # Main script
└── results/               # Output directory
```

## Customization

### Bring Your Own Traces

Edit `run_crafter_verifier.py` to load your traces:

```python
# Load custom traces
traces = load_traces_from_file("my_traces.jsonl")

# Build dataset
dataset = {
    "name": "my_verifier_calibration",
    "task_description": "Evaluate my agent traces",
    "examples": [
        {
            "id": t["trace_id"],
            "input": {"trace": t["trace"], "trace_id": t["trace_id"]},
            "expected": {"score": t["gold_score"]},
        }
        for t in traces
    ],
}

# Pass to config
config.dataset = dataset
```

### Different Scoring Aspects

Tell the proposer what aspects to evaluate:

```toml
[graph_optimization]
topology_guidance = """
Build a DAG with:
1. Early nodes to digest raw trace data (actions, observations)
2. Aspect scoring nodes (task completion, safety, efficiency)
3. Final aggregation node to produce overall score
"""
```

## Troubleshooting

### Low Agreement with Gold

1. Check gold scores are normalized (0-1)
2. Increase generations for more exploration
3. Use stronger proposer model: `model = "gpt-4.1"`

### Graph Produces Wrong Output Format

1. Ensure `graph_type = "verifier"` (not "policy")
2. Check `output_schema` includes `score` field
3. Review proposer prompts for format guidance

