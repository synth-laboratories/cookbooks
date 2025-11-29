"""In-Process Task App Example.

This script demonstrates running a task app in-process with the SDK,
eliminating the need to run a separate server or tunnel.

Usage:
    python in_process.py

Requires:
    - SYNTH_API_KEY environment variable
    - OPENAI_API_KEY environment variable (for LLM calls)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    """Run prompt learning with in-process task app."""
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
    from synth_ai.sdk.task.in_process import InProcessTaskApp

    # Verify environment
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY not set")
        sys.exit(1)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not set (needed for LLM calls)")
        sys.exit(1)

    # Task app path
    task_app_path = Path(__file__).parent.parent / "task_app.py"
    if not task_app_path.exists():
        print(f"Error: Task app not found: {task_app_path}")
        sys.exit(1)

    # Config path
    config_path = Path(__file__).parent.parent / "configs" / "gepa.toml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    print("Starting in-process task app...")

    # Run task app in-process
    async with InProcessTaskApp(
        task_app_path=task_app_path,
        port=8114,  # Use different port to avoid conflicts
        env={
            "ENVIRONMENT_API_KEY": "in-process-key",
            "OPENAI_API_KEY": openai_key,
        },
    ) as task_app:
        print(f"Task app running at: {task_app.url}")

        # Create job with in-process task app URL
        job = PromptLearningJob.from_config(
            config_path=config_path,
            overrides={
                "task_url": task_app.url,
            },
        )

        # Override the ENVIRONMENT_API_KEY for this job
        os.environ["ENVIRONMENT_API_KEY"] = "in-process-key"

        print("Submitting job...")

        try:
            job_id = job.submit()
            print(f"Job submitted: {job_id}")
        except Exception as e:
            print(f"Error: {e}")
            return

        print("Polling for completion...")

        # Poll until complete
        result = await asyncio.to_thread(
            job.poll_until_complete,
            timeout=3600.0,
        )

        print("\n=== Results ===")
        print(f"Status: {result.get('status', 'unknown')}")

        if result.get("status") == "completed":
            results = job.get_results()
            print(f"Best Score: {results.get('best_score', 'N/A')}")

    print("\nTask app stopped. Done!")


if __name__ == "__main__":
    asyncio.run(main())
