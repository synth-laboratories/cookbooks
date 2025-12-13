# HotpotQA Policy Graph Optimization

Optimize a multi-step reasoning graph for HotpotQA multi-hop question answering.

## What You'll Build

A **policy graph** that:
1. Receives a question + supporting passages
2. Performs multi-hop reasoning across passages
3. Extracts the answer + supporting facts

```
[question, context] → [reasoning] → [parse_output] → {answer, support}
```

## Quick Start

```bash
# Set API keys
export SYNTH_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# Run optimization
python run_hotpotqa.py
```

## Expected Output

```
🚀 Starting HotpotQA Policy Graph Optimization
============================================================
   Dataset: hotpotqa
   Graph type: policy
   Structure: dag
   Generations: 5
   Seeds: 20

📊 Streaming optimization progress...
   [73201c89] score=0.813
[Gen 1] best=0.813, archive=1
   [0ba51190] score=0.786
   [204e19b3] score=0.813
[Gen 2] best=0.813, archive=3
   ...
[Gen 5] best=0.843, archive=1

✅ Optimization complete!
   Best score: 0.843
   Generations: 5
   Candidates: 13

📊 Best Graph (prompts + UML):
--------------------------------------------------
Prompts (per node):

--- reasoning ---
You are an AI assistant solving multi-hop questions.
Question: <input>question</input>
Context: <input>context</input>

Analyze the passages to find the answer...

Dataflow (UML):
@startuml
start
reasoning --> parse_output
parse_output --> end
@enduml
--------------------------------------------------

📄 Saved to results/hotpotqa_result_20251212.json
```

## How It Works

1. **Initial Generation**: LLM creates a complete graph from task description
2. **Evaluation**: Graph runs on training seeds, scored by LLM judge
3. **Evolution**: Proposer mutates prompts/structure based on feedback
4. **Pareto Selection**: Non-dominated candidates preserved for diversity
5. **Output**: Best graph export (prompts + dataflow)

## Configuration

See `config.toml`:

```toml
[graph_optimization]
dataset_name = "hotpotqa"
graph_type = "policy"
graph_structure = "dag"

[graph_optimization.proposer]
model = "gpt-4.1"

[graph_optimization.population]
num_generations = 5
children_per_generation = 3

[graph_optimization.evaluation]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

[graph_optimization.limits]
max_spend_usd = 10.0
```

## Graph Structure Options

| Structure | Description | Best For |
|-----------|-------------|----------|
| `single_prompt` | One LLM node | Simple Q&A |
| `dag` | Linear pipeline | Multi-step reasoning |
| `conditional` | Branching logic | Complex routing |

## Scoring

The optimization uses **LLM-as-judge** with per-task rubrics:

- **1.0**: Exact/semantic match with expected answer
- **0.7**: Partial match (contains correct answer + extra info)
- **0.3**: Shows understanding but wrong answer
- **0.0**: Completely wrong or missing

## Files

```
hotpotqa/
├── README.md           # This file
├── config.toml         # GEPA configuration
├── run_hotpotqa.py     # Main script
└── results/            # Output directory
```

## Customization

### Change Model

Edit `config.toml`:
```toml
[graph_optimization.proposer]
model = "gpt-4.1"  # or "gpt-4o", "claude-3-sonnet"
```

### Add More Seeds

```toml
[graph_optimization.evaluation]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

### Topology Guidance

Tell the proposer what kind of graph you want:
```toml
[graph_optimization]
topology_guidance = "Use a single LLM node with chain-of-thought prompting"
```

## Troubleshooting

### Low Scores

1. Increase generations: `num_generations = 10`
2. Use stronger model: `model = "gpt-4.1"`
3. Add more seeds for better signal

### Slow Convergence

1. Start with `dag` structure (simpler than `conditional`)
2. Reduce children: `children_per_generation = 2`
3. Check proposer feedback in logs

