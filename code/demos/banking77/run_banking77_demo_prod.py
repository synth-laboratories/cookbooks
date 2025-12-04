#!/usr/bin/env python3
"""
Production version of Banking77 GEPA Demo.

This script runs the Banking77 GEPA optimization against the production backend.
It uses Cloudflare quick tunnels to expose the local task app to the production backend.

The script will automatically:
- Detect if backend is localhost or remote
- Create a Cloudflare quick tunnel if backend is remote
- Verify tunnel connectivity

Usage:
    cd /Users/joshpurtell/Documents/GitHub/research
    BACKEND_URL=https://agent-learning.onrender.com SYNTH_API_KEY=sk_live_... uv run --directory /Users/joshpurtell/Documents/GitHub/research python /Users/joshpurtell/Documents/GitHub/cookbooks/demos/banking77/run_banking77_demo_prod.py

Or set environment variables:
    export BACKEND_URL=https://agent-learning.onrender.com
    export SYNTH_API_KEY=sk_live_...
    uv run --directory /Users/joshpurtell/Documents/GitHub/research python /Users/joshpurtell/Documents/GitHub/cookbooks/demos/banking77/run_banking77_demo_prod.py

For quick test with tiny budget:
    export BACKEND_URL=https://agent-learning.onrender.com
    export SYNTH_API_KEY=sk_live_...
    export TEST_MODE=1
    uv run --directory /Users/joshpurtell/Documents/GitHub/research python /Users/joshpurtell/Documents/GitHub/cookbooks/demos/banking77/run_banking77_demo_prod.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add synth-ai to path
synth_ai_root = Path(__file__).parent.parent.parent.parent / "synth-ai"
if synth_ai_root.exists():
    sys.path.insert(0, str(synth_ai_root))

# Import from the dev version (same directory)
# Add parent directory to path to allow relative import
sys.path.insert(0, str(Path(__file__).parent))
from run_banking77_demo import (
    Banking77DemoTracker,
    save_comprehensive_results,
)

# Import SDK components
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.task.in_process import InProcessTaskApp


async def run():
    """Run Banking77 GEPA demo against production backend."""
    # Production backend configuration
    backend_url = os.getenv("BACKEND_URL") or "https://agent-learning.onrender.com"
    # For named tunnels, SYNTH_API_KEY is required (not just ENVIRONMENT_API_KEY)
    api_key = os.getenv("SYNTH_API_KEY")
    
    if not api_key:
        raise ValueError(
            "SYNTH_API_KEY must be set for production backend.\n"
            "Set it with: export SYNTH_API_KEY=sk_live_..."
        )

    # Configuration paths
    script_dir = Path(__file__).parent
    # Always use full config
    config_path = script_dir / "banking77_gepa_demo.toml"
    print("📊 Using FULL config (20 candidates, 40 pareto seeds, 200 val seeds)")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Task app path - try multiple locations
    # Try cookbooks dev first (same repo)
    task_app_path = Path(__file__).parent.parent.parent / "dev" / "task_apps" / "banking77" / "banking77_task_app.py"
    
    if not task_app_path.exists():
        # Try research repo
        research_root = Path(__file__).parent.parent.parent.parent / "research"
        task_app_path = research_root / "walkthroughs" / "gepa" / "task_app" / "banking77_task_app.py"
    
    if not task_app_path.exists():
        raise FileNotFoundError(
            f"Task app not found. Tried:\n"
            f"  - {Path(__file__).parent.parent.parent / 'dev' / 'task_apps' / 'banking77' / 'banking77_task_app.py'}\n"
            f"  - {research_root / 'walkthroughs' / 'gepa' / 'task_app' / 'banking77_task_app.py'}"
        )

    print("=" * 80)
    print("🚀 Banking77 GEPA Demo - PRODUCTION")
    print("=" * 80)
    print(f"Backend URL: {backend_url}")
    print(f"Config: {config_path}")
    print(f"Task App: {task_app_path}")
    print("=" * 80)
    print()

    # Determine tunnel mode based on backend URL
    # - localhost/127.0.0.1 → "local" (no tunnel needed, backend can reach local task app)
    # - remote backend → "named" (use managed Cloudflare tunnel)
    is_backend_localhost = (
        backend_url.startswith("http://localhost")
        or backend_url.startswith("http://127.0.0.1")
    )
    
    if is_backend_localhost:
        tunnel_mode = "local"
        print("ℹ️  Local mode: Backend is localhost, task app will use localhost (no tunnel)")
    else:
        tunnel_mode = "named"  # Use managed Cloudflare tunnel (more reliable than quick tunnels)
        print("ℹ️  Remote mode: Backend is remote, using named/managed Cloudflare tunnel")
    
    task_app_api_key = api_key

    # Set environment variables for task app
    os.environ["ENVIRONMENT_API_KEY"] = task_app_api_key
    if os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    if os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Start InProcessTaskApp with appropriate tunnel mode
    print(f"🔧 Starting InProcessTaskApp with tunnel_mode={tunnel_mode}...")
    if tunnel_mode == "named":
        print("   Using managed Cloudflare tunnel (requires SYNTH_API_KEY).")
    async with InProcessTaskApp(
        task_app_path=str(task_app_path),
        port=8118,
        auto_find_port=True,
        api_key=task_app_api_key,
        tunnel_mode=tunnel_mode,
        health_check_timeout=120.0,
        skip_tunnel_verification=True,  # Skip local DNS verification - backend uses different DNS
        force_new_tunnel=True,  # Create fresh tunnel, bypasses stale tunnels
    ) as task_app:
        print(f"✅ InProcessTaskApp started at {task_app.url}")
        print(f"   Tunnel URL: {task_app.url}")
        
        # Give tunnel a moment to stabilize
        print("⏳ Waiting for tunnel to stabilize...")
        await asyncio.sleep(3)

        overrides = {
            "task_url": task_app.url,
            "prompt_learning.task_app_url": task_app.url,
        }

        job = PromptLearningJob.from_config(
            config_path=str(config_path),
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            overrides=overrides,
        )

        print(f"Submitting GEPA job to {backend_url}...")
        try:
            job_id = job.submit()
            print(f"✓ Job submitted: {job_id}")
        except Exception as e:
            print(f"❌ Failed to submit job: {e}")
            print(f"   Task app URL: {task_app.url}")
            print(f"   This might be a tunnel connectivity issue. Try again in a moment.")
            raise

        # Create tracker
        tracker = Banking77DemoTracker()

        # Stream events via SSE with periodic job status checks (like redactle example)
        async def stream_events():
            import httpx
            from synth_ai.sdk.api.train.utils import ensure_api_base

            api_base = ensure_api_base(backend_url)
            events_stream_url = f"{api_base}/prompt-learning/online/jobs/{job_id}/events/stream"
            job_url = f"{api_base}/prompt-learning/online/jobs/{job_id}"

            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0)) as client:
                buffer = ""
                events_received = 0
                last_status_check = time.time()
                status_check_interval = 10.0  # Check job status every 10 seconds
                
                try:
                    print("🔌 Connecting to SSE stream...")
                    async with client.stream(
                        "GET",
                        f"{events_stream_url}?since_seq=0",
                        headers={"X-API-Key": api_key, "Accept": "text/event-stream"},
                        timeout=httpx.Timeout(None, connect=30.0),
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            raise Exception(f"SSE stream failed: {response.status_code} - {error_text.decode('utf-8', errors='ignore')}")
                        
                        print("✅ SSE stream connected")
                        
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            
                            buffer += chunk.decode("utf-8", errors="ignore")
                            
                            # Process complete events (SSE uses double newline as delimiter)
                            while "\n\n" in buffer:
                                event_block, buffer = buffer.split("\n\n", 1)
                                if not event_block.strip():
                                    continue
                                
                                # Parse SSE event
                                event_data = {}
                                event_type = None
                                for line in event_block.split("\n"):
                                    if line.startswith("event:"):
                                        event_type = line[6:].strip()
                                    elif line.startswith("data:"):
                                        try:
                                            event_data = json.loads(line[5:].strip())
                                        except json.JSONDecodeError:
                                            continue
                                
                                if event_data:
                                    events_received += 1
                                    # Get event type from payload if not in SSE header
                                    if not event_type:
                                        event_type = event_data.get("type", "unknown")
                                    
                                    tracker.update_from_event({"type": event_type, "data": event_data.get("data", event_data)})
                                    
                                    # Print progress
                                    if events_received == 1:
                                        print(f"📊 Received first event: {event_type}")
                                    elif events_received % 10 == 0:
                                        print(f"📊 Received {events_received} events...")
                                    
                                    # Check for completion
                                    if event_type in ("prompt.learning.completed", "prompt.learning.gepa.complete"):
                                        print(f"✅ Job completed via SSE: {event_type}")
                                        tracker.status = "complete"
                                        return
                                    
                                    # Check for failure
                                    if event_type in ("prompt.learning.failed", "prompt.learning.error"):
                                        print(f"❌ Job failed via SSE: {event_type}")
                                        tracker.status = "failed"
                                        tracker.finish_reason = event_data.get("message", "Unknown error")
                                        return
                            
                            # Periodically check job status via API (in case SSE misses completion)
                            current_time = time.time()
                            if current_time - last_status_check >= status_check_interval:
                                try:
                                    resp = await client.get(
                                        job_url,
                                        headers={"X-API-Key": api_key},
                                        timeout=5.0,
                                    )
                                    if resp.status_code == 200:
                                        job_data = resp.json()
                                        status = job_data.get("status", "")
                                        if status in ("succeeded", "completed", "failed", "cancelled"):
                                            print(f"✅ Job finished (detected via polling): {status}")
                                            tracker.status = "complete" if status in ("succeeded", "completed") else "failed"
                                            return
                                except Exception:
                                    pass  # Ignore polling errors, continue with SSE
                                last_status_check = current_time
                        
                        print(f"⚠️  SSE stream closed (received {events_received} events), checking job status...")
                        
                except httpx.TimeoutException:
                    print(f"⚠️  SSE connection timeout, falling back to polling...")
                except Exception as e:
                    print(f"⚠️  SSE stream error: {e}")
                    print("   Falling back to polling...")

        # Stream events (with timeout protection)
        try:
            await asyncio.wait_for(stream_events(), timeout=600.0)  # 10 minute max for SSE
        except asyncio.TimeoutError:
            print("⚠️  SSE stream timed out after 10 minutes, checking job status...")

        # Check job status and poll if needed
        print("📊 Checking job status...")
        try:
            result = await asyncio.wait_for(asyncio.to_thread(job.get_status), timeout=30.0)
            status = result.get("status") if isinstance(result, dict) else result
            print(f"📊 Job status: {status}")
        except asyncio.TimeoutError:
            print("⚠️ Job status check timed out, assuming succeeded from polling")
            status = "succeeded" if tracker.status == "complete" else "failed"
            result = {"status": status}

        if status not in ("succeeded", "completed", "failed", "cancelled"):
            print("⏳ Job still running, polling until completion...")
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(job.poll_until_complete, timeout=300.0, interval=5.0),
                    timeout=600.0
                )
                status = result.get("status") if isinstance(result, dict) else result
                print(f"✅ Job finished with status: {status}")
            except asyncio.TimeoutError:
                print("⚠️ Polling timed out, continuing with results fetch...")
                status = "succeeded"
                result = {"status": status}

        if status == "failed":
            try:
                error_msg = await asyncio.wait_for(
                    asyncio.to_thread(lambda: job.get_results().get("error", "Unknown error")),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                error_msg = "Timeout fetching error"
            print(f"❌ Job failed: {error_msg}")
            tracker.status = "failed"
            tracker.finish_reason = error_msg

        # Fetch results
        print("📊 Fetching results...")
        from synth_ai.sdk.api.train.utils import ensure_api_base
        from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient

        api_base = ensure_api_base(backend_url)
        client = PromptLearningClient(api_base, api_key)
        prompt_results = await client.get_prompts(job_id)
        scoring_summary = await client.get_scoring_summary(job_id)

        # Get job results
        try:
            job_results = await asyncio.wait_for(asyncio.to_thread(job.get_results), timeout=60.0)
        except asyncio.TimeoutError:
            print("⚠️ Fetching job results timed out, using empty results")
            job_results = {}

        # Extract baseline_score from job_results (authoritative source)
        tracker.baseline_score = (
            tracker.baseline_score  # Keep if captured from events
            or job_results.get("baseline_score")
            or scoring_summary.get("baseline_score")
            or scoring_summary.get("baseline_accuracy")
        )

        # Extract candidates from job_results (authoritative source, like redactle example)
        print("📊 Extracting candidates from job_results...")
        attempted_candidates = job_results.get("attempted_candidates", [])
        optimized_candidates = job_results.get("optimized_candidates", [])
        
        # Combine and deduplicate by version_id (like redactle)
        all_candidates_dict = {}
        for cand in attempted_candidates:
            version_id = cand.get("version_id") or cand.get("id") or f"unknown_{len(all_candidates_dict)}"
            all_candidates_dict[version_id] = cand
        
        # Overwrite with optimized candidates (they have more complete data)
        for cand in optimized_candidates:
            version_id = cand.get("version_id") or cand.get("id") or f"unknown_{len(all_candidates_dict)}"
            all_candidates_dict[version_id] = cand
        
        # Convert to tracker format
        tracker.candidates = []
        for cand in all_candidates_dict.values():
            try:
                prompt_text = tracker._extract_prompt_text_from_candidate(cand)
            except RecursionError:
                # If recursion error, try simpler extraction
                prompt_text = (
                    cand.get("prompt_text")
                    or (cand.get("object", {}) or {}).get("prompt_text")
                    or str(cand.get("best_prompt", ""))[:500] if cand.get("best_prompt") else None
                )
                print(f"⚠️  Recursion error extracting prompt for candidate {cand.get('version_id')}, using fallback")
            
            candidate = {
                "candidate_id": cand.get("version_id") or cand.get("id") or f"c{len(tracker.candidates)}",
                "accuracy": cand.get("accuracy") or cand.get("score") or cand.get("train_accuracy", 0),
                "val_accuracy": cand.get("val_accuracy") or cand.get("full_score"),
                "train_accuracy": cand.get("train_accuracy") or cand.get("accuracy"),
                "prompt_text": prompt_text,
                "generation": cand.get("generation"),
                "parent_id": cand.get("parent_id"),
                "is_pareto": cand.get("is_pareto", False),
                "timestamp": time.time(),
                "raw_data": cand,
            }
            tracker.candidates.append(candidate)
            acc = candidate["accuracy"]
            if acc and isinstance(acc, (int, float)) and acc > tracker.best_score:
                tracker.best_score = acc
        
        print(f"✓ Extracted {len(tracker.candidates)} candidates from job_results")

        # Extract usage data
        tracker.usage_data = job_results.get("usage") or scoring_summary.get("usage") or {}

        # Save comprehensive results
        output_dir = script_dir / "results_prod"
        print(f"💾 Saving comprehensive results to {output_dir}...")
        try:
            save_comprehensive_results(
                output_dir=output_dir,
                tracker=tracker,
                job_results=job_results,
                prompt_results=prompt_results,
                scoring_summary=scoring_summary,
            )
            print("✅ Results saved successfully!")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            import traceback
            traceback.print_exc()

        # Print summary
        print()
        print("=" * 80)
        print("📊 FINAL SUMMARY")
        print("=" * 80)
        print(f"Job ID: {job_id}")
        print(f"Status: {tracker.status}")
        print(f"Baseline Score: {tracker.baseline_score}")
        print(f"Best Score: {tracker.best_score}")
        print(f"Candidates Found: {len(tracker.candidates)}")
        if tracker.usage_data:
            cost = tracker.usage_data.get("cost_usd", 0)
            print(f"Total Cost: ${cost:.2f}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run())

