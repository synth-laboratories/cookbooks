# Graph Learning Cookbook

Complete guide to **Graph GEPA** - optimizing graph-based policies and verifiers using genetic evolution.

## What You'll Build

- **Policy Graphs**: Multi-step reasoning pipelines that solve tasks (e.g., HotpotQA)
- **Verifier Graphs**: Judges that evaluate agent traces (e.g., Crafter game evaluation)
- Automated evolution from scratch → optimized graph

## Quick Start

```bash
# HotpotQA: Optimize a policy graph for multi-hop QA
cd hotpotqa && python run_hotpotqa.py

# Crafter: Calibrate a verifier graph for game trace evaluation
cd crafter && python run_crafter_verifier.py
```

## What is Graph GEPA?

Graph GEPA evolves **YAML-based graph programs** using:
1. **LLM Proposer**: Generates initial graphs from task descriptions
2. **Patch-based Evolution**: Mutates prompts, adds/removes nodes
3. **Pareto Archive**: Keeps diverse non-dominated candidates
4. **Instance-wise Dominance**: Seed-by-seed comparison for robustness

Unlike prompt optimization (which tunes a single prompt), Graph GEPA creates **multi-node pipelines** with:
- LLM reasoning nodes
- Python parsing nodes
- Conditional routing
- DAG execution

## Graph Types

| Type | Purpose | Example |
|------|---------|---------|
| **Policy** | Solve tasks | HotpotQA: question → reasoning → answer |
| **Verifier** | Judge outputs | Crafter: trace → aspects → score |

### Policy Graphs

Used to **solve tasks**. Takes input (question, context) and produces output (answer).

```
[question, context] → [reasoning] → [parse_output] → [answer]
```

### Verifier Graphs

Used to **evaluate agent performance**. Takes trace data and produces a judgment score.

```
[trace] → [analyze_aspects] → [combine_scores] → [score, reasoning]
```

## Prerequisites

- Python 3.11+
- `synth-ai` SDK: `pip install synth-ai`
- API keys: `SYNTH_API_KEY`, `OPENAI_API_KEY`
- Backend URL (cloud or local): `SYNTH_BACKEND_URL`

## Project Structure

```
graph_learning/
├── README.md                # This file
├── hotpotqa/                # Policy graph for multi-hop QA
│   ├── README.md            # HotpotQA-specific guide
│   ├── config.toml          # Graph GEPA configuration
│   ├── run_hotpotqa.py      # Main script
│   └── results/             # Output files
├── crafter/                 # Verifier graph for game traces
│   ├── README.md            # Crafter-specific guide
│   ├── config.toml          # Graph GEPA configuration
│   ├── dataset.json         # Sample trace data
│   ├── run_crafter_verifier.py
│   └── results/             # Output files
└── sdk/                     # Reusable client code
    └── graph_gepa_client.py # API client
```

## API Endpoints

Graph GEPA uses these authenticated endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/graph-gepa/jobs` | POST | Start optimization job |
| `/graph-gepa/jobs/{id}/events` | GET (SSE) | Stream progress events |
| `/graph-gepa/jobs/{id}/result` | GET | Get final result |
| `/graph-gepa/jobs/{id}` | DELETE | Cancel job |

## Configuration Reference

```toml
[graph_optimization]
dataset_name = "hotpotqa"      # Dataset identifier
graph_type = "policy"          # "policy" or "verifier"
graph_structure = "dag"        # "single_prompt", "dag", or "conditional"

[graph_optimization.proposer]
model = "gpt-4.1"              # LLM for graph generation
temperature = 0.7

[graph_optimization.population]
num_generations = 5
children_per_generation = 3

[graph_optimization.evaluation]
seeds = [0, 1, 2, 3, 4]        # Training seeds

[graph_optimization.limits]
max_spend_usd = 10.0
timeout_seconds = 1800
```

## Example Output

After optimization, you receive a **public graph export** (prompts + UML):

```
Prompts (per node):

--- reasoning ---
You are an AI assistant solving multi-hop questions.
Question: <input>question</input>
Context: <input>context</input>

Analyze the context and provide your answer...

Dataflow (UML):
@startuml
start
reasoning --> parse_output
parse_output --> end
end
@enduml
```

> **Security Note**: Raw YAML is never exposed. You receive prompts + dataflow only.

## Cookbook Examples

### [HotpotQA Policy Graph](hotpotqa/)

Optimize a graph that answers multi-hop questions:
- Input: Question + supporting passages
- Output: Answer + supporting facts
- Graph: reasoning → parsing → output

### [Crafter Verifier Graph](crafter/)

Calibrate a judge for game agent traces:
- Input: Game execution trace
- Output: Score (0-1) + reasoning
- Graph: aspect analysis → score combination

## See Also

- [Prompt Learning Cookbook](../prompt_learning/) - Single-prompt optimization
- [SDK Documentation](https://docs.usesynth.ai/sdk/training/graph-optimization)
- [API Reference](https://docs.usesynth.ai/api/graph-gepa)

