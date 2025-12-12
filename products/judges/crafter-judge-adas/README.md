# Crafter Judge‑ADAS Cookbook (Graph Judges)

This cookbook shows how to **discover a judge graph for Crafter** using
**Judge‑ADAS** (Graph‑GEPA + gold labels).

We:
1. Run real Crafter rollouts to collect **v3 traces** with event rewards.
2. Compute a gold **outcome score based only on achievement count** and store it as an outcome reward.
3. Convert traces → an ADAS‑style dataset (trace → gold score).
4. Evolve a **judge graph** that matches the gold scores, and also emits rich feedback.

> Important: Graph YAML/spec is proprietary.  
> This workflow never asks you to publish or commit YAML. If you want to share a judge,
> use the public `.txt` export routes.

---

## Prerequisites

You need a local Crafter task app and v3 trace DB running.

From the monorepo root:

```bash
cd backend
./start_crafter_with_sqld.sh
```

This starts:
- Crafter service at `http://localhost:8901`
- sqld/libsql trace DB at `traces/v3/crafter.db`

You also need a Synth install that includes `synth_ai.lm` (the full agent runtime),
plus your API keys for Qwen inference.

---

## 1) Collect gold Crafter traces

Run **100 real episodes** with a Qwen policy model (`qwen/qwen3-32b`):

```bash
cd monorepo
uv run python backend/graphs/gepa_integration/tests/judge_demo_crafter_collect_traces.py \
  --episodes 100 \
  --policy-model qwen/qwen3-32b \
  --difficulty easy \
  --max-turns 100
```

Outputs (local only):
- `traces/judge_demo/crafter_gold_labels.json`
- `traces/judge_demo/crafter_gold_traces.jsonl`
- `traces/judge_demo/crafter_judge_adas_dataset.json`

Each trace includes:
- full v3 `event_history`
- event rewards from environment steps
- a gold **outcome reward** in `[0,1]` derived from achievements only

---

## 2) Run Judge‑ADAS to discover a judge graph

Now evolve a judge that matches those gold scores.

The discovered judge is richer than our achievement‑only baselines:
- It predicts an outcome `score` matching gold.
- It emits **numeric event‑level rewards** (`event_rewards`) per turn.
- It provides **text feedback** for both outcome and event rewards.

```bash
cd monorepo
uv run python backend/graphs/gepa_integration/tests/test_judge_adas_crafter.py \
  --generations 5 \
  --children 3
```

The script prints train/val correlation metrics and writes a summary to:

`traces/judge_demo/crafter_judge_demo_result.json`

Do **not** commit any `best_yaml` produced by the run.

---

## 3) Use the discovered judge

If you want to score new traces, persist the best judge to your org via:

```bash
curl -X POST http://localhost:8000/graph-judge/graphs \
  -H "Authorization: Bearer $SYNTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "crafter_judge_demo",
    "yaml_content": "<best_yaml from your local run>",
    "description": "Crafter judge discovered via Judge‑ADAS",
    "gepa_job_id": "<job_id from the run>"
  }'
```

Then score a trace:

```bash
curl -X POST http://localhost:8000/graph-judge/score \
  -H "Authorization: Bearer $SYNTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "graph_name": "crafter_judge_demo",
    "trace": <your_v3_trace_json>
  }'
```

To share the judge publicly, download the **redacted export**:

```bash
curl -L http://localhost:8000/graph-judge/graphs/<graph_id>/export.txt \
  -H "Authorization: Bearer $SYNTH_API_KEY"
```

That `.txt` is the only supported public artifact.
