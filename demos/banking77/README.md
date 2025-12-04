# Banking77 GEPA Demo

A comprehensive demonstration of GEPA (Genetic Evolutionary Prompt Adaptation) optimization on the Banking77 intent classification task.

## Overview

This demo runs a full GEPA optimization with:
- **Large pareto set**: 100 seeds for pareto scoring
- **Large validation**: 200 seeds for validation evaluation
- **15 candidates**: Enough rollouts/trials to generate and evaluate 15 candidates
- **Comprehensive data collection**: All necessary data for in-depth analysis

## Configuration

The demo uses `banking77_gepa_demo.toml` with the following key settings:

- **Train seeds**: 120 seeds (100 for pareto, 20 for feedback)
- **Validation seeds**: 200 seeds from test split
- **Rollout budget**: 500 rollouts
- **Population**: 12 initial candidates, 16 generations, 6 children per generation
- **Archive**: Max 15 candidates, pareto set size 100

## Running the Demo

```bash
# Set environment variables
export BACKEND_URL=http://localhost:8000
export SYNTH_API_KEY=your_api_key
export ENVIRONMENT_API_KEY=your_api_key

# Run the demo
cd cookbooks/demos/banking77
python run_banking77_demo.py
```

## Output Files

All results are saved to `results/` directory:

### `candidates.json`
- All candidates with their scores (train and validation)
- Prompts for each candidate
- Generation and parent information
- Pareto status

### `per_seed_scores.json`
- Per-seed scores for each candidate
- Shows which seeds improved and which resisted improvement
- Useful for analyzing seed-specific performance

### `confusion_matrices.json`
- Confusion matrices for each candidate (if rollout trajectory data is available)
- Shows which intents are confused with which
- Helps identify systematic errors

### `pareto_frontier.json`
- Top 10 Pareto-optimal candidates
- Shows the best candidates across different metrics

### `optimistic_scores.json`
- Baseline score
- Best achieved score
- Optimistic score (best possible improvement)
- Improvement potential analysis

### `cost_and_time.json`
- Total time (seconds and minutes)
- Cost breakdown by category
- Token usage statistics
- Finish reason

### `analysis_report.md`
- Comprehensive markdown report
- Summary statistics
- Top 10 candidates with prompts
- File descriptions

## Analysis

The saved data enables rich analysis including:

1. **Progress Analysis**: Which candidates improved over baseline
2. **Seed Analysis**: Which seeds got better and which resisted improvement
3. **Confusion Patterns**: Which intent pairs are commonly confused
4. **Pareto Analysis**: Trade-offs between different metrics
5. **Cost Analysis**: Cost per candidate, cost per improvement

## Example Analysis Workflow

1. Load `candidates.json` to see all candidates and their scores
2. Load `per_seed_scores.json` to analyze seed-specific improvements
3. Load `confusion_matrices.json` to identify systematic errors
4. Load `pareto_frontier.json` to see optimal candidates
5. Use `cost_and_time.json` for cost-benefit analysis

## Notes

- The demo uses an in-process task app, so no external deployment is needed
- Results are saved incrementally, so you can analyze partial results if the run is interrupted
- Confusion matrices require rollout trajectory data, which may not always be available via the API

