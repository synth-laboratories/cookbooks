#!/usr/bin/env python3
"""
GEPA Demo Cookbook - Run GEPA optimization on the Crafter environment

This cookbook demonstrates how to use the GEPA (Generative Evolution of Prompt
Architectures) demo API to optimize prompts for the Crafter game environment.

The demo runs on Synth's production infrastructure and requires no authentication.

Usage:
    python gepa_demo.py

    # Or with custom options:
    python gepa_demo.py --nickname "MyRun" --generations 2 --population 3

For more information, see: https://www.usesynth.ai/blog/gepa-for-agents
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx

# =============================================================================
# Configuration
# =============================================================================

# Production API endpoint
API_BASE = "https://agent-learning.onrender.com"
BLOG_SLUG = "gepa-for-agents"

# Available policy models (routed via Synth interceptor)
AVAILABLE_MODELS = [
    "openai/gpt-oss-20b",     # Default - faster
    "openai/gpt-oss-120b",    # Larger model
    "qwen/qwen3-32b",         # Qwen3
    "moonshotai/kimi-k2-instruct",  # Kimi K2
]


@dataclass
class DemoConfig:
    """Configuration for a GEPA demo run."""
    num_generations: int = 2  # 1-5 generations
    population_size: int = 3  # 2-10 candidates per generation
    max_steps_per_rollout: int = 10  # 5-50 steps per game
    policy_model: str = "openai/gpt-oss-20b"


@dataclass
class DemoStatus:
    """Status of a running demo."""
    demo_id: str
    status: str
    progress: float
    current_generation: int
    total_generations: int
    best_score: Optional[float]
    candidates_evaluated: int
    phase: Optional[str]


# =============================================================================
# API Client
# =============================================================================

def start_demo(
    nickname: Optional[str] = None,
    config: Optional[DemoConfig] = None,
) -> dict:
    """Start a new GEPA demo.

    Args:
        nickname: Optional nickname for leaderboard attribution
        config: Demo configuration (uses defaults if not provided)

    Returns:
        Response dict with demo_id and stream_url

    Raises:
        httpx.HTTPStatusError: If the request fails
    """
    config = config or DemoConfig()

    payload = {
        "environment": "crafter",
        "config": {
            "num_generations": config.num_generations,
            "population_size": config.population_size,
            "max_steps_per_rollout": config.max_steps_per_rollout,
            "policy_model": config.policy_model,
        },
    }

    if nickname:
        payload["nickname"] = nickname[:32]  # Max 32 chars

    response = httpx.post(
        f"{API_BASE}/api/blog/{BLOG_SLUG}/demos",
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def get_demo_status(demo_id: str) -> Optional[DemoStatus]:
    """Get the current status of a demo.

    Args:
        demo_id: The demo ID returned from start_demo

    Returns:
        DemoStatus if found, None otherwise
    """
    response = httpx.get(
        f"{API_BASE}/api/blog/{BLOG_SLUG}/demos/{demo_id}",
        timeout=10.0,
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()
    data = response.json()

    return DemoStatus(
        demo_id=data.get("demo_id", demo_id),
        status=data.get("status", "unknown"),
        progress=data.get("progress", 0.0),
        current_generation=data.get("current_generation", 0),
        total_generations=data.get("total_generations", 0),
        best_score=data.get("best_score"),
        candidates_evaluated=data.get("candidates_evaluated", 0),
        phase=data.get("phase"),
    )


def list_active_demos() -> list[dict]:
    """List all currently active demos.

    Returns:
        List of active demo info dicts
    """
    response = httpx.get(
        f"{API_BASE}/api/blog/{BLOG_SLUG}/demos",
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("demos", [])


def stream_demo_events(demo_id: str, timeout: float = 1200.0):
    """Stream SSE events from a running demo.

    Args:
        demo_id: The demo ID to stream events from
        timeout: Maximum time to wait for events (default 20 minutes)

    Yields:
        Tuples of (event_type, event_data)
    """
    url = f"{API_BASE}/api/blog/{BLOG_SLUG}/demos/{demo_id}/stream"

    with httpx.stream("GET", url, timeout=timeout) as response:
        response.raise_for_status()

        event_type = None
        event_data = []

        for line in response.iter_lines():
            line = line.strip()

            if not line:
                # Empty line = end of event
                if event_type and event_data:
                    data_str = "\n".join(event_data)
                    try:
                        parsed_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        parsed_data = data_str
                    yield event_type, parsed_data
                event_type = None
                event_data = []
                continue

            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                event_data.append(line[5:].strip())


# =============================================================================
# Demo Runner
# =============================================================================

def run_demo(
    nickname: Optional[str] = None,
    num_generations: int = 2,
    population_size: int = 3,
    policy_model: str = "openai/gpt-oss-20b",
    verbose: bool = True,
) -> dict:
    """Run a complete GEPA demo and wait for results.

    Args:
        nickname: Optional nickname for leaderboard
        num_generations: Number of optimization generations (1-5)
        population_size: Candidates per generation (2-10)
        policy_model: Model to use for the agent
        verbose: Print progress updates

    Returns:
        Final results dict with best candidate info
    """
    config = DemoConfig(
        num_generations=num_generations,
        population_size=population_size,
        policy_model=policy_model,
    )

    if verbose:
        print(f"Starting GEPA demo...")
        print(f"  Generations: {num_generations}")
        print(f"  Population: {population_size}")
        print(f"  Model: {policy_model}")
        if nickname:
            print(f"  Nickname: {nickname}")
        print()

    # Start the demo
    try:
        result = start_demo(nickname=nickname, config=config)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print("Error: A demo is already running. Please wait for it to complete.")
            print("You can watch it at: https://www.usesynth.ai/blog/gepa-for-agents")
            sys.exit(1)
        raise

    demo_id = result["demo_id"]
    stream_url = result["stream_url"]

    if verbose:
        print(f"Demo started: {demo_id}")
        print(f"Stream URL: {API_BASE}{stream_url}")
        print(f"Watch live at: https://www.usesynth.ai/blog/gepa-for-agents")
        print()

    # Track state
    best_score = None
    best_candidate = None
    final_results = {}

    # Stream events
    try:
        for event_type, data in stream_demo_events(demo_id):
            if event_type == "demo.started":
                if verbose:
                    print("Demo running...")

            elif event_type == "generation.started":
                gen = data.get("generation", 0)
                total = data.get("total_generations", 0)
                if verbose:
                    print(f"\nGeneration {gen}/{total}")

            elif event_type == "candidate.new":
                if verbose:
                    cid = data.get("candidate_id", "?")
                    print(f"  New candidate: {cid}")

            elif event_type == "rollout.complete":
                score = data.get("score")
                cid = data.get("candidate_id")
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    best_candidate = cid
                if verbose:
                    seed = data.get("seed", "?")
                    print(f"    Rollout seed={seed}: score={score:.3f}" if score else f"    Rollout seed={seed}: failed")

            elif event_type == "demo.progress":
                progress = data.get("progress", 0)
                phase = data.get("phase", "")
                rollouts = data.get("rollouts_completed", 0)
                total_rollouts = data.get("rollouts_total", 0)
                if verbose and rollouts > 0:
                    print(f"  Progress: {progress:.1%} ({rollouts}/{total_rollouts} rollouts, {phase})")

            elif event_type == "demo.complete":
                if verbose:
                    print(f"\nDemo complete!")
                final_results = data
                break

            elif event_type == "demo.error":
                error_msg = data.get("error", "Unknown error")
                print(f"\nDemo error: {error_msg}")
                sys.exit(1)

            elif event_type == "heartbeat":
                # Ignore heartbeats
                pass

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        sys.exit(0)

    # Print final results
    if verbose and final_results:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Best Score: {final_results.get('best_score', 'N/A')}")
        print(f"Candidates Evaluated: {final_results.get('candidates_evaluated', 'N/A')}")
        print(f"Total Rollouts: {final_results.get('total_rollouts', 'N/A')}")

        best = final_results.get("best_candidate", {})
        if best:
            print(f"\nBest Candidate: {best.get('candidate_id', 'N/A')}")
            print(f"  Generation: {best.get('generation', 'N/A')}")

            # Print the optimized prompt (truncated)
            stages = best.get("stages", {})
            for stage_id, stage in stages.items():
                instruction = stage.get("instruction", stage.get("content", ""))
                if instruction:
                    preview = instruction[:200] + "..." if len(instruction) > 200 else instruction
                    print(f"\n  [{stage_id.upper()}]:")
                    print(f"    {preview}")

        print(f"\nView on leaderboard: https://www.usesynth.ai/blog/gepa-for-agents")

    return final_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run a GEPA demo to optimize Crafter game prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gepa_demo.py
  python gepa_demo.py --nickname "MyExperiment"
  python gepa_demo.py --generations 3 --population 5
  python gepa_demo.py --model "openai/gpt-oss-120b"

Available models:
  - openai/gpt-oss-20b (default, faster)
  - openai/gpt-oss-120b (larger)
  - qwen/qwen3-32b
  - moonshotai/kimi-k2-instruct

For more info: https://www.usesynth.ai/blog/gepa-for-agents
        """
    )

    parser.add_argument(
        "--nickname", "-n",
        type=str,
        help="Your nickname for the leaderboard (max 32 chars)"
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=2,
        choices=range(1, 6),
        metavar="1-5",
        help="Number of optimization generations (default: 2)"
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=3,
        choices=range(2, 11),
        metavar="2-10",
        help="Population size per generation (default: 3)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openai/gpt-oss-20b",
        choices=AVAILABLE_MODELS,
        help="Policy model for the agent (default: openai/gpt-oss-20b)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List currently active demos and exit"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # List active demos
    if args.list:
        demos = list_active_demos()
        if not demos:
            print("No active demos")
        else:
            print(f"Active demos ({len(demos)}):")
            for demo in demos:
                status = demo.get("status", "?")
                progress = demo.get("progress", 0)
                nick = demo.get("nickname", "Anonymous")
                print(f"  {demo['demo_id']}: {status} ({progress:.0%}) - {nick}")
        return

    # Run demo
    run_demo(
        nickname=args.nickname,
        num_generations=args.generations,
        population_size=args.population,
        policy_model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
