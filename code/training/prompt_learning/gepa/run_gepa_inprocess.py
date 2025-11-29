#!/usr/bin/env python3
"""
In-Process GEPA Demo for Banking77
===================================

This script demonstrates running GEPA optimization with a task app started
entirely in-process - no separate terminals or manual process management needed!

Everything happens in a single Python script:
1. Task app starts in a background thread
2. Cloudflare tunnel opens automatically
3. GEPA job runs using the tunnel URL
4. Everything cleans up automatically on exit

Usage:
    cd /Users/joshpurtell/Documents/GitHub/cookbooks
    source .env
    uv run python code/training/prompt_learning/gepa/run_gepa_inprocess.py

Requirements:
    - SYNTH_API_KEY in .env
    - GROQ_API_KEY in .env
    - ENVIRONMENT_API_KEY in .env
    - cloudflared binary (will auto-install if missing)

Configuration:
    Default: Uses Render backend (agent-learning.onrender.com)
    Override: Set BACKEND_BASE_URL env var to use different backend

    The script automatically matches tunnel mode:
    - If BACKEND_BASE_URL is localhost → both backend and task app use localhost (local/local)
    - If BACKEND_BASE_URL is a remote URL → task app uses Cloudflare tunnel
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment from repo root
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
env_path = repo_root / ".env"
load_dotenv(env_path)

from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.task import InProcessTaskApp


async def main():
    """Run GEPA optimization with in-process task app."""

    print("\n" + "=" * 80)
    print("In-Process GEPA Demo - Banking77 Intent Classification")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env")
        sys.exit(1)

    if not os.getenv("SYNTH_API_KEY"):
        print("❌ Error: SYNTH_API_KEY required in .env")
        sys.exit(1)

    # Configuration paths
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.toml"

    # Task app is in dev/task_apps/banking77/
    task_app_path = repo_root / "dev" / "task_apps" / "banking77" / "banking77_task_app.py"

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    if not task_app_path.exists():
        print(f"❌ Error: Task app not found: {task_app_path}")
        sys.exit(1)

    # Default to Render backend (more stable than Modal for prompt learning)
    backend_url = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com/api")
    api_key = os.getenv("SYNTH_API_KEY")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test-key")

    # Determine tunnel mode based on backend URL
    is_backend_localhost = (
        backend_url.startswith("http://localhost")
        or backend_url.startswith("http://127.0.0.1")
    )

    if is_backend_localhost:
        # Backend is localhost → use local mode for task app (no tunnel)
        os.environ["SYNTH_TUNNEL_MODE"] = "local"
        use_local_mode = True
        print("ℹ️  Configuration: local/local")
        print("   Backend: localhost")
        print("   Task App: localhost (no tunnel)")
    else:
        # Backend is remote → use tunnel mode for task app
        os.environ["SYNTH_TUNNEL_MODE"] = "quick"
        use_local_mode = False
        os.environ["EXTERNAL_BACKEND_URL"] = backend_url.rstrip("/")
        print("ℹ️  Configuration: tunnel/tunnel")
        print(f"   Backend: {backend_url}")
        print(f"   Task app: will create Cloudflare tunnel")

    print("\nConfiguration:")
    print(f"  Config: {config_path}")
    print(f"  Task App: {task_app_path}")
    print(f"  Backend: {backend_url}")
    print(f"  API Key: {api_key[:15]}...")
    print()

    # Run GEPA with in-process task app
    try:
        async with InProcessTaskApp(
            task_app_path=task_app_path,
            port=8114,
            api_key=task_app_api_key,
        ) as task_app:
            print(f"✅ Task app running at: {task_app.url}")
            print(f"✅ Cloudflare tunnel active")
            print()

            # Create GEPA job
            print("=" * 80)
            print("Running GEPA Optimization")
            print("=" * 80 + "\n")

            # Override task_app_url with the tunnel URL
            overrides = {
                "prompt_learning": {
                    "task_app_url": task_app.url,
                    "results_folder": "/tmp/gepa_banking77_results",
                }
            }

            job = PromptLearningJob.from_config(
                config_path=config_path,
                backend_url=backend_url,
                api_key=api_key,
                task_app_api_key=task_app_api_key,
                overrides=overrides,
            )

            print(f"Submitting job to backend: {backend_url}")
            print(f"Task app tunnel URL: {task_app.url}")
            print()

            job_id = job.submit()
            print(f"✅ Job submitted: {job_id}\n")

            # Poll for completion
            start_time = time.time()
            last_status = None

            def on_status(status):
                nonlocal last_status
                elapsed = time.time() - start_time
                state = status.get("status", "unknown")

                # Only print if status changed or every 10 seconds
                if state != last_status or int(elapsed) % 10 == 0:
                    timestamp = time.strftime("%H:%M:%S")
                    progress = status.get("progress", {})
                    if progress:
                        completed = progress.get("completed", 0)
                        total = progress.get("total", 0)
                        if total > 0:
                            pct = (completed / total) * 100
                            print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state} ({completed}/{total} = {pct:.1f}%)")
                        else:
                            print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}")
                    else:
                        print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}")
                    last_status = state

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: job.poll_until_complete(
                    timeout=3600.0,
                    interval=5.0,
                    on_status=on_status,
                ),
            )

            total_time = time.time() - start_time
            print(f"\n✅ GEPA optimization complete in {total_time:.1f}s\n")

            # Get results
            from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
            from synth_ai.sdk.api.train.utils import ensure_api_base

            client = PromptLearningClient(
                ensure_api_base(backend_url),
                api_key,
            )
            prompt_results = await client.get_prompts(job._job_id)

            print("=" * 80)
            print("Results")
            print("=" * 80 + "\n")

            if prompt_results.best_score is not None:
                print(f"Best score: {prompt_results.best_score:.2%}")
            else:
                print("Best score: N/A (job may have failed)")

            if prompt_results.attempted_candidates is not None:
                print(f"Total candidates: {prompt_results.attempted_candidates}")
            print()

            if prompt_results.best_prompt:
                print("Best prompt:")
                print("-" * 80)
                # Extract prompt text
                if "prompt_sections" in prompt_results.best_prompt:
                    sections = prompt_results.best_prompt["prompt_sections"]
                    prompt_text = "\n\n".join(
                        [s.get("content", "") for s in sections if s.get("content")]
                    )
                    print(prompt_text[:500])
                    if len(prompt_text) > 500:
                        print("\n... [truncated]")
                print()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("✅ In-process GEPA demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
