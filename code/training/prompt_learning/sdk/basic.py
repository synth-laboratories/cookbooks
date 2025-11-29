"""Basic Prompt Learning SDK Example.

This script demonstrates the simplest SDK workflow for prompt optimization:
1. Load config from TOML file
2. Submit job
3. Poll until complete
4. Extract results

Usage:
    python basic.py

Requires:
    - SYNTH_API_KEY environment variable
    - ENVIRONMENT_API_KEY environment variable
    - Task app running (python ../task_app.py)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """Run basic prompt learning workflow."""
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

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
    config_path = Path(__file__).parent.parent / "configs" / "gepa.toml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")

    # Create job from config
    job = PromptLearningJob.from_config(
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
        print("Make sure task app is running: python ../task_app.py")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Job submission failed: {e}")
        sys.exit(1)

    print("Polling for completion (this may take several minutes)...")

    # Poll until complete
    result = job.poll_until_complete(timeout=3600.0)

    print("\n=== Results ===")
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get("status") == "completed":
        # Get detailed results
        results = job.get_results()

        print(f"Best Score: {results.get('best_score', 'N/A')}")

        best_prompt = results.get("best_prompt")
        if best_prompt:
            print("\nBest Prompt:")
            for section in best_prompt.get("sections", []):
                role = section.get("role", "unknown")
                content = section.get("content", "")[:200]  # Truncate for display
                print(f"  [{role.upper()}]: {content}...")
    else:
        print(f"Job did not complete successfully: {result}")


if __name__ == "__main__":
    main()
