"""Advanced RL SDK Example.

This script demonstrates advanced SDK features:
1. Custom status callbacks for progress monitoring
2. Metrics extraction and analysis
3. Checkpoint management
4. Error handling and retry logic

Usage:
    python advanced.py

Requires:
    - SYNTH_API_KEY environment variable
    - ENVIRONMENT_API_KEY environment variable
    - Task app running (python ../task_app.py or use heartdisease app)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any


def on_status_update(status: dict[str, Any]) -> None:
    """Handle status updates during polling."""
    job_status = status.get("status", "unknown")
    progress = status.get("progress", {})

    current = progress.get("current_rollout", 0)
    total = progress.get("total_rollouts", 0)

    if total > 0:
        pct = (current / total) * 100
        print(f"  Status: {job_status} | Progress: {current}/{total} ({pct:.1f}%)")
    else:
        print(f"  Status: {job_status}")


def analyze_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Analyze training metrics and return summary."""
    steps = metrics.get("steps", [])
    if not steps:
        return {"error": "No steps in metrics"}

    rewards = [s.get("reward", 0) for s in steps]
    losses = [s.get("loss", 0) for s in steps if s.get("loss") is not None]

    analysis = {
        "total_steps": len(steps),
        "reward": {
            "min": min(rewards),
            "max": max(rewards),
            "mean": sum(rewards) / len(rewards),
            "final": rewards[-1] if rewards else 0,
        },
    }

    if losses:
        analysis["loss"] = {
            "min": min(losses),
            "max": max(losses),
            "mean": sum(losses) / len(losses),
            "final": losses[-1],
        }

    # Calculate improvement
    if len(rewards) >= 10:
        early_avg = sum(rewards[:10]) / 10
        late_avg = sum(rewards[-10:]) / 10
        analysis["improvement"] = {
            "early_avg": early_avg,
            "late_avg": late_avg,
            "delta": late_avg - early_avg,
            "relative": (late_avg - early_avg) / max(early_avg, 0.001) * 100,
        }

    return analysis


def main() -> None:
    """Run advanced RL workflow with monitoring."""
    from synth_ai.sdk.api.train.rl import RLJob

    # Verify environment
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY not set")
        sys.exit(1)

    env_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not env_key:
        print("Error: ENVIRONMENT_API_KEY not set")
        sys.exit(1)

    # Config path
    config_path = Path(__file__).parent.parent / "configs" / "rl.toml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    print("=" * 50)

    # Create job from config
    job = RLJob.from_config(config_path=config_path)

    print("Submitting job...")
    start_time = time.time()

    # Submit with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            job_id = job.submit()
            print(f"Job submitted: {job_id}")
            break
        except ValueError as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Error: Task app health check failed after {max_retries} attempts")
                print("Make sure task app is running:")
                print("  python -m synth_ai.sdk.task.server --app heartdisease --port 8114")
                sys.exit(1)
        except RuntimeError as e:
            print(f"Error: Job submission failed: {e}")
            sys.exit(1)

    print("\nPolling for completion with status updates...")
    print("-" * 50)

    # Poll with custom callback
    result = job.poll_until_complete(
        timeout=7200.0,
        interval=10.0,
        on_status=on_status_update,
    )

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    status = result.get("status", "unknown")
    print(f"Status: {status}")

    if status == "completed":
        # Get and analyze metrics
        metrics = job.get_metrics()
        analysis = analyze_metrics(metrics)

        print(f"\nTraining Steps: {analysis.get('total_steps', 0)}")

        if "reward" in analysis:
            r = analysis["reward"]
            print(f"\nReward Statistics:")
            print(f"  Min:   {r['min']:.4f}")
            print(f"  Max:   {r['max']:.4f}")
            print(f"  Mean:  {r['mean']:.4f}")
            print(f"  Final: {r['final']:.4f}")

        if "loss" in analysis:
            l = analysis["loss"]
            print(f"\nLoss Statistics:")
            print(f"  Min:   {l['min']:.4f}")
            print(f"  Max:   {l['max']:.4f}")
            print(f"  Mean:  {l['mean']:.4f}")
            print(f"  Final: {l['final']:.4f}")

        if "improvement" in analysis:
            imp = analysis["improvement"]
            print(f"\nImprovement Analysis:")
            print(f"  Early Avg (first 10):  {imp['early_avg']:.4f}")
            print(f"  Late Avg (last 10):    {imp['late_avg']:.4f}")
            print(f"  Absolute Improvement:  {imp['delta']:.4f}")
            print(f"  Relative Improvement:  {imp['relative']:.1f}%")

        # Check for checkpoints
        checkpoints = result.get("checkpoints", [])
        if checkpoints:
            print(f"\nCheckpoints: {len(checkpoints)}")
            for cp in checkpoints[-3:]:  # Show last 3
                print(f"  - Step {cp.get('step', '?')}: {cp.get('path', 'N/A')}")

        # Get final model if available
        model_id = result.get("model_id")
        if model_id:
            print(f"\nTrained Model ID: {model_id}")
            print("Use this model for inference with InferenceClient")

    elif status == "failed":
        error = result.get("error", "Unknown error")
        print(f"\nJob failed: {error}")

        # Check for partial metrics
        try:
            metrics = job.get_metrics()
            if metrics and metrics.get("steps"):
                print(f"\nPartial training completed: {len(metrics['steps'])} steps")
        except Exception:
            pass

    else:
        print(f"\nUnexpected status: {result}")


if __name__ == "__main__":
    main()
