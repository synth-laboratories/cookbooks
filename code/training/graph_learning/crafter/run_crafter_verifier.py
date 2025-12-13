#!/usr/bin/env python3
"""Crafter Verifier Graph Calibration.

Calibrates a verifier graph to judge Crafter game agent traces.

Usage:
    python run_crafter_verifier.py
    python run_crafter_verifier.py --traces my_traces.jsonl
    python run_crafter_verifier.py --backend-url http://localhost:8000

Requires:
    - SYNTH_API_KEY environment variable
    - OPENAI_API_KEY environment variable (for LLM judge)

Install:
    pip install synth-ai
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Load .env file if available
try:
    from dotenv import load_dotenv
    for env_path in [
        Path(__file__).parent.parent.parent.parent.parent.parent.parent / "synth-ai" / ".env",
        Path(__file__).parent / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

# Import from synth-ai SDK (try both paths for flexibility)
try:
    # When running via `cd synth-ai && uv run python ...`
    from products.graph_gepa import GraphOptimizationClient, GraphOptimizationConfig
    from products.graph_gepa.config import (
        GraphType, GraphStructure, ProposerConfig, EvolutionConfig, SeedsConfig, LimitsConfig
    )
except ImportError as e1:
    try:
        # When synth-ai is installed as a package
        from synth_ai.products.graph_gepa import GraphOptimizationClient, GraphOptimizationConfig
        from synth_ai.products.graph_gepa.config import (
            GraphType, GraphStructure, ProposerConfig, EvolutionConfig, SeedsConfig, LimitsConfig
        )
    except ImportError as e2:
        print("❌ Error: synth-ai not installed.")
        print(f"   products error: {e1}")
        print(f"   synth_ai error: {e2}")
        print("   Run from synth-ai dir: cd /path/to/synth-ai && uv run python this_script.py")
        print("   Or install synth-ai: pip install synth-ai")
        sys.exit(1)


def load_traces_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load traces from JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def build_adas_dataset(traces: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Build ADAS-format dataset for verifier calibration.
    
    The backend expects:
      - tasks: list of {"task_id": str, "input": dict}
      - gold_outputs: list of {"task_id": str, "output": dict}
      - metadata: optional dict with task_description etc.
    
    Args:
        traces: List of trace dicts with 'trace', 'trace_id', 'gold_score'
        name: Dataset name
        
    Returns:
        ADAS-format dataset dict
    """
    tasks = []
    gold_outputs = []
    
    for i, t in enumerate(traces):
        task_id = t.get("trace_id", f"trace_{i}")
        gold_score = t.get("gold_score", 0.0)
        achievements = t.get("achievements_count", 0)
        
        # Task input
        tasks.append({
            "task_id": task_id,
            "input": {
                "trace": t.get("trace", {}),
                "trace_id": task_id,
            },
        })
        
        # Gold output for this task
        gold_outputs.append({
            "task_id": task_id,
            "output": {
                "score": gold_score,
                "achievements_count": achievements,
            },
        })
    
    return {
        "tasks": tasks,
        "gold_outputs": gold_outputs,
        "metadata": {
            "name": name,
            "task_description": (
                "Evaluate Crafter game agent execution traces. "
                "Score based on survival, exploration, achievement progress, and efficiency."
            ),
            "input_schema": {"trace": "object", "trace_id": "string"},
            "output_schema": {"score": "number", "reasoning": "string"},
        },
    }


def get_sample_traces() -> List[Dict[str, Any]]:
    """Return sample traces for demo purposes."""
    return [
        {
            "trace_id": "demo_trace_1",
            "trace": {
                "total_steps": 100,
                "achievements": [],
                "final_health": 0,
                "events": [{"action": "move", "result": "moved_forward"}],
            },
            "gold_score": 0.0,
            "achievements_count": 0,
        },
        {
            "trace_id": "demo_trace_2",
            "trace": {
                "total_steps": 500,
                "achievements": ["collect_wood", "make_wood_pickaxe"],
                "final_health": 5,
                "events": [{"action": "chop_tree", "result": "collected_wood"}],
            },
            "gold_score": 0.09,
            "achievements_count": 2,
        },
        {
            "trace_id": "demo_trace_3",
            "trace": {
                "total_steps": 1000,
                "achievements": ["collect_wood", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe"],
                "final_health": 9,
                "events": [{"action": "mine", "result": "collected_stone"}],
            },
            "gold_score": 0.18,
            "achievements_count": 4,
        },
    ]


async def main():
    """Run Crafter verifier calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crafter Verifier Graph Calibration")
    parser.add_argument("--traces", default=None, help="Path to traces JSONL file")
    parser.add_argument("--backend-url", default=os.environ.get("SYNTH_BACKEND_URL", "http://localhost:8000"))
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--children", type=int, default=2)
    args = parser.parse_args()
    
    # Check API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("❌ Error: SYNTH_API_KEY not set")
        sys.exit(1)
    
    # Load traces
    if args.traces:
        print(f"📂 Loading traces from {args.traces}")
        traces = load_traces_from_jsonl(args.traces)
    else:
        print("📂 Using sample traces (pass --traces for your own data)")
        traces = get_sample_traces()
    
    print(f"   Loaded {len(traces)} traces")
    
    # Validate traces
    assert len(traces) > 0, "❌ No traces loaded! Provide --traces or check sample data."
    
    # Build inline dataset
    dataset = build_adas_dataset(traces, "crafter_verifier_calibration")
    
    # Validate dataset structure (critical - 0 data is a non-starter)
    assert "tasks" in dataset, "❌ Dataset missing 'tasks' key"
    assert "gold_outputs" in dataset, "❌ Dataset missing 'gold_outputs' key"
    assert len(dataset["tasks"]) > 0, f"❌ Dataset has 0 tasks! Expected {len(traces)}"
    assert len(dataset["gold_outputs"]) > 0, f"❌ Dataset has 0 gold outputs! Expected {len(traces)}"
    assert len(dataset["tasks"]) == len(traces), f"❌ Task count mismatch: {len(dataset['tasks'])} != {len(traces)}"
    assert len(dataset["gold_outputs"]) == len(traces), f"❌ Gold output count mismatch: {len(dataset['gold_outputs'])} != {len(traces)}"
    
    print(f"📦 Dataset: {len(dataset['tasks'])} tasks, {len(dataset['gold_outputs'])} gold outputs")
    
    # Build config using synth-ai SDK
    config = GraphOptimizationConfig(
        dataset_name="crafter_verifier_calibration",
        dataset=dataset,  # Inline upload
        graph_type=GraphType.VERIFIER,
        graph_structure=GraphStructure.DAG,
        task_description=dataset["metadata"]["task_description"],
        input_schema={"trace": "object", "trace_id": "string"},
        output_schema={"score": "number", "reasoning": "string", "sub_scores": "object"},
        topology_guidance="Build a DAG with early nodes to digest trace aspects (survival, exploration, achievements, efficiency) and a final node to combine into overall score.",
        proposer=ProposerConfig(model="gpt-4.1"),
        evolution=EvolutionConfig(num_generations=args.generations, children_per_generation=args.children),
        seeds=SeedsConfig(train=list(range(len(traces)))),
        limits=LimitsConfig(max_spend_usd=20.0),
        scoring_strategy="rubric",
        judge_model="gpt-4o-mini",
    )
    
    print()
    print("=" * 60)
    print("🎮 Crafter Verifier Graph Calibration")
    print("=" * 60)
    print(f"   Traces: {len(traces)}")
    print(f"   Graph type: {config.graph_type.value}")
    print(f"   Structure: {config.graph_structure.value}")
    print(f"   Generations: {config.evolution.num_generations}")
    print(f"   Backend: {args.backend_url}")
    print()
    
    # Run optimization
    async with GraphOptimizationClient(base_url=args.backend_url, api_key=api_key) as client:
        print("📡 Starting job...")
        try:
            job_id = await client.start_job(config)
            print(f"✅ Job started: {job_id}")
        except Exception as e:
            print(f"❌ Failed to start job: {e}")
            sys.exit(1)
        
        print("\n📊 Streaming optimization progress...")
        best_score = 0.0
        try:
            async for event in client.stream_events(job_id):
                event_type = event.get("type", "")
                data = event.get("data", {})
                
                if event_type == "candidate_evaluated":
                    cid = data.get("candidate_id", "?")[:8]
                    score = data.get("score", 0)
                    print(f"   [{cid}] score={score:.3f}")
                elif event_type == "generation_completed":
                    gen = data.get("generation", 0)
                    best = data.get("best_score", 0)
                    archive = data.get("archive_size", 0)
                    print(f"[Gen {gen}] best={best:.3f}, archive={archive}")
                    best_score = max(best_score, best)
                elif event_type == "job_completed":
                    best_score = data.get("best_score", best_score)
                    print(f"\n✅ Calibration complete: best={best_score:.3f}")
                    break
                elif event_type == "job_failed":
                    print(f"\n❌ Job failed: {data.get('error')}")
                    sys.exit(1)
        except Exception as e:
            print(f"\n⚠️  Streaming interrupted: {e}")
        
        # Get result
        print("\n📥 Fetching result...")
        result = await client.get_result(job_id)
        
        print("\n" + "=" * 60)
        print("✅ Calibration Complete!")
        print("=" * 60)
        print(f"   Best score: {result.get('best_score', 0):.3f}")
        print(f"   Generations: {result.get('generations_completed', 0)}")
        print(f"   Candidates: {result.get('total_candidates_evaluated', 0)}")
        
        # Show graph export
        export = result.get("best_graph_export")
        if export:
            print("\n📊 Best Graph (prompts + UML):")
            print("-" * 50)
            print(export[:2000])
            if len(export) > 2000:
                print(f"... ({len(export) - 2000} more chars)")
            print("-" * 50)
        
        # Save result
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"crafter_verifier_result_{timestamp}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n📄 Saved to {out_path}")
        
        print("\n" + "=" * 60)
        print("✅ Done!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
