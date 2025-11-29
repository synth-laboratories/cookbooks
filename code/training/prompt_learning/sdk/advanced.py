"""Advanced Prompt Learning SDK Example.

This script demonstrates advanced SDK features:
1. Custom polling with status callbacks
2. Error handling and retries
3. Result extraction and analysis
4. Comparing GEPA vs MIPRO algorithms

Usage:
    python advanced.py

Requires:
    - SYNTH_API_KEY environment variable
    - ENVIRONMENT_API_KEY environment variable
    - Task app running (python ../task_app.py)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


def on_status_update(status: dict[str, Any]) -> None:
    """Callback for status updates during polling."""
    current_status = status.get("status", "unknown")
    progress = status.get("progress", {})

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Status: {current_status}", end="")

    if "current_trial" in progress:
        print(f" | Trial: {progress['current_trial']}/{progress.get('total_trials', '?')}", end="")
    if "best_score" in progress:
        print(f" | Best: {progress['best_score']:.4f}", end="")

    print()


def run_optimization(config_name: str) -> dict[str, Any]:
    """Run a single optimization job and return results."""
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

    config_path = Path(__file__).parent.parent / "configs" / config_name

    print(f"\n{'='*60}")
    print(f"Running {config_name}")
    print(f"{'='*60}")

    job = PromptLearningJob.from_config(config_path=config_path)

    # Submit with error handling
    try:
        job_id = job.submit()
        print(f"Job ID: {job_id}")
    except Exception as e:
        print(f"Submission failed: {e}")
        return {"status": "failed", "error": str(e)}

    # Poll with status callback
    try:
        result = job.poll_until_complete(
            timeout=3600.0,
            interval=10.0,
            on_status=on_status_update,
        )
    except TimeoutError:
        print("Job timed out!")
        return {"status": "timeout", "job_id": job_id}

    # Extract results
    if result.get("status") == "completed":
        try:
            detailed = job.get_results()
            return {
                "status": "completed",
                "job_id": job_id,
                "best_score": detailed.get("best_score"),
                "best_prompt": detailed.get("best_prompt"),
                "top_prompts": detailed.get("top_prompts", [])[:3],
                "num_candidates": len(detailed.get("attempted_candidates", [])),
            }
        except Exception as e:
            print(f"Failed to extract results: {e}")
            return {"status": "completed", "job_id": job_id, "error": str(e)}
    else:
        return {"status": result.get("status", "unknown"), "job_id": job_id}


def compare_algorithms() -> None:
    """Compare GEPA and MIPRO on the same task."""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON: GEPA vs MIPRO")
    print("="*60)

    results = {}

    # Run GEPA
    results["gepa"] = run_optimization("gepa.toml")

    # Run MIPRO
    results["mipro"] = run_optimization("mipro.toml")

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    for algo, data in results.items():
        print(f"\n{algo.upper()}:")
        print(f"  Status: {data.get('status')}")
        if "best_score" in data:
            print(f"  Best Score: {data['best_score']:.4f}")
        if "num_candidates" in data:
            print(f"  Candidates Evaluated: {data['num_candidates']}")
        if "error" in data:
            print(f"  Error: {data['error']}")

    # Determine winner
    gepa_score = results.get("gepa", {}).get("best_score", 0)
    mipro_score = results.get("mipro", {}).get("best_score", 0)

    if gepa_score and mipro_score:
        print(f"\n{'='*60}")
        if gepa_score > mipro_score:
            print(f"WINNER: GEPA ({gepa_score:.4f} > {mipro_score:.4f})")
        elif mipro_score > gepa_score:
            print(f"WINNER: MIPRO ({mipro_score:.4f} > {gepa_score:.4f})")
        else:
            print(f"TIE: Both algorithms achieved {gepa_score:.4f}")

    # Save results
    output_path = Path(__file__).parent.parent / "results" / "comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def extract_and_save_prompt() -> None:
    """Extract the best prompt and save it for production use."""
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

    config_path = Path(__file__).parent.parent / "configs" / "gepa.toml"

    print("\nRunning optimization to extract production prompt...")

    job = PromptLearningJob.from_config(config_path=config_path)
    job_id = job.submit()
    print(f"Job ID: {job_id}")

    result = job.poll_until_complete(timeout=3600.0)

    if result.get("status") != "completed":
        print(f"Job failed: {result.get('status')}")
        return

    # Get the best prompt
    results = job.get_results()
    best_prompt = results.get("best_prompt")

    if not best_prompt:
        print("No best prompt found in results")
        return

    # Format for production use
    production_prompt = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "job_id": job_id,
        "best_score": results.get("best_score"),
        "prompt": best_prompt,
        "usage": {
            "system_message": next(
                (s["content"] for s in best_prompt.get("sections", []) if s.get("role") == "system"),
                None
            ),
        },
    }

    # Save to file
    output_path = Path(__file__).parent.parent / "results" / "production_prompt.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(production_prompt, f, indent=2)

    print(f"\nProduction prompt saved to: {output_path}")
    print(f"Best score: {results.get('best_score'):.4f}")

    if production_prompt["usage"]["system_message"]:
        print("\nSystem message (first 200 chars):")
        print(f"  {production_prompt['usage']['system_message'][:200]}...")


def main() -> None:
    """Run advanced examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Prompt Learning SDK Examples")
    parser.add_argument(
        "--mode",
        choices=["compare", "extract", "single"],
        default="single",
        help="Mode: compare algorithms, extract prompt, or single run",
    )
    parser.add_argument(
        "--config",
        default="gepa.toml",
        help="Config file for single mode (default: gepa.toml)",
    )

    args = parser.parse_args()

    # Verify environment
    if not os.environ.get("SYNTH_API_KEY"):
        print("Error: SYNTH_API_KEY not set")
        sys.exit(1)
    if not os.environ.get("ENVIRONMENT_API_KEY"):
        print("Error: ENVIRONMENT_API_KEY not set")
        sys.exit(1)

    if args.mode == "compare":
        compare_algorithms()
    elif args.mode == "extract":
        extract_and_save_prompt()
    else:
        result = run_optimization(args.config)
        print(f"\nFinal result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
