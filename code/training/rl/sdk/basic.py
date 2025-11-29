"""Basic RL SDK Example.

This script demonstrates the simplest SDK workflow for RL training:
1. Load config from TOML file
2. Submit job (requires running task app)
3. Poll until complete
4. Extract results

Usage:
    python basic.py

Requires:
    - SYNTH_API_KEY environment variable
    - ENVIRONMENT_API_KEY environment variable
    - Task app running (python ../task_app.py or use heartdisease app)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    """Run basic RL workflow."""
    from synth_ai.sdk.api.train.rl import RLJob

    # Verify environment
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY not set")
        print("Set it with: export SYNTH_API_KEY=your-key")
        sys.exit(1)

    env_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not env_key:
        print("Error: ENVIRONMENT_API_KEY not set")
        print("Set it with: export ENVIRONMENT_API_KEY=your-env-key")
        sys.exit(1)

    # Config path
    config_path = Path(__file__).parent.parent / "configs" / "rl.toml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")

    # Create job from config
    job = RLJob.from_config(
        config_path=config_path,
        # Optional: override backend URL for local development
        # backend_url="http://localhost:8000/api",
    )

    print("Submitting job...")

    # Submit job
    try:
        job_id = job.submit()
        print(f"Job submitted: {job_id}")
    except ValueError as e:
        print(f"Error: Task app health check failed: {e}")
        print("Make sure task app is running:")
        print("  python -m synth_ai.sdk.task.server --app heartdisease --port 8114")
        print("Or:")
        print("  python ../task_app.py")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Job submission failed: {e}")
        sys.exit(1)

    print("Polling for completion (this may take several minutes)...")

    # Poll until complete
    result = job.poll_until_complete(timeout=7200.0)

    print("\n=== Results ===")
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get("status") == "completed":
        # Get training metrics
        metrics = job.get_metrics()

        final_reward = result.get("final_reward", "N/A")
        print(f"Final Reward: {final_reward}")

        if metrics and "steps" in metrics:
            print(f"Training Steps: {len(metrics['steps'])}")

            # Show reward progression
            rewards = [s.get("reward", 0) for s in metrics["steps"]]
            if rewards:
                print(f"Reward Range: {min(rewards):.4f} - {max(rewards):.4f}")
                print(f"Average Reward: {sum(rewards) / len(rewards):.4f}")
    else:
        print(f"Job did not complete successfully: {result}")


if __name__ == "__main__":
    main()
