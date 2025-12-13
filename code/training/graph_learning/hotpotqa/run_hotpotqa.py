#!/usr/bin/env python3
"""HotpotQA Policy Graph Optimization.

Optimizes a multi-step reasoning graph for HotpotQA multi-hop question answering.

Usage:
    python run_hotpotqa.py
    python run_hotpotqa.py --backend-url http://localhost:8000
    python run_hotpotqa.py --config custom_config.toml

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
except ImportError:
    try:
        # When synth-ai is installed as a package
        from synth_ai.products.graph_gepa import GraphOptimizationClient, GraphOptimizationConfig
    except ImportError:
        print("❌ Error: synth-ai not installed.")
        print("   Run from synth-ai dir: cd /path/to/synth-ai && uv run python this_script.py")
        print("   Or install synth-ai: pip install synth-ai")
        sys.exit(1)


async def main():
    """Run HotpotQA policy graph optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HotpotQA Policy Graph Optimization")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.toml"),
        help="Path to config TOML file",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("SYNTH_BACKEND_URL", "http://localhost:8000"),
        help="Backend API URL",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results"),
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    # Check API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("❌ Error: SYNTH_API_KEY not set")
        print("   Set it with: export SYNTH_API_KEY=your-key")
        sys.exit(1)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config not found: {config_path}")
        sys.exit(1)
    
    config = GraphOptimizationConfig.from_toml(str(config_path))
    
    print("=" * 60)
    print("🚀 Starting HotpotQA Policy Graph Optimization")
    print("=" * 60)
    print(f"   Dataset: {config.dataset_name}")
    print(f"   Graph type: {config.graph_type.value}")
    print(f"   Structure: {config.graph_structure.value}")
    print(f"   Generations: {config.evolution.num_generations}")
    print(f"   Seeds: {len(config.seeds.train)}")
    print(f"   Backend: {args.backend_url}")
    print()
    
    # Run optimization
    async with GraphOptimizationClient(base_url=args.backend_url, api_key=api_key) as client:
        # Start job
        print("📡 Starting job...")
        try:
            job_id = await client.start_job(config)
            print(f"✅ Job started: {job_id}")
        except Exception as e:
            print(f"❌ Failed to start job: {e}")
            sys.exit(1)
        
        # Stream events
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
                    print(f"\n✅ Optimization complete: best={best_score:.3f}")
                    break
                
                elif event_type == "job_failed":
                    print(f"\n❌ Job failed: {data.get('error', 'Unknown error')}")
                    sys.exit(1)
        
        except Exception as e:
            print(f"\n⚠️  Streaming interrupted: {e}")
        
        # Get result
        print("\n📥 Fetching result...")
        result = await client.get_result(job_id)
        
        print("\n" + "=" * 60)
        print("✅ Optimization Complete!")
        print("=" * 60)
        print(f"   Best score: {result.get('best_score', 0):.3f}")
        print(f"   Generations: {result.get('generations_completed', 0)}")
        print(f"   Candidates: {result.get('total_candidates_evaluated', 0)}")
        print(f"   Cost: ${result.get('total_cost_usd', 0):.4f}")
        print(f"   Duration: {result.get('duration_seconds', 0):.1f}s")
        
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
        out_path = out_dir / f"hotpotqa_result_{timestamp}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n📄 Saved to {out_path}")
        
        print("\n" + "=" * 60)
        print("✅ Done!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
