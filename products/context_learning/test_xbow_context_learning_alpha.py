#!/usr/bin/env python3
"""Context Learning (Alpha) XBOW CTF integration cookbook.

This is a runnable cookbook "test" showing end-to-end Context Learning
against the XBOW Codex/CTF task app in `research/customers/xbow/task_app`.

It:
1. Launches the XBOW task app locally on a free port
2. Submits a Context Learning job (alpha-only)
3. Streams progress via SSE until completion
4. Downloads the best preflight script

Usage:
    # From repo root
    uv run python cookbooks/products/context_learning/test_xbow_context_learning_alpha.py

Requirements:
    - SYNTH_API_KEY (alpha tier)
    - OPENAI_API_KEY (for GEPA mutation LLM if your backend needs it)
    - Optional: ENVIRONMENT_API_KEY if task app auth is enabled
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(url: str, timeout_s: float = 60.0) -> None:
    import urllib.request
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Task app not healthy at {url}: {last_err}")


def _simple_toml(data: dict, prefix: str = "") -> str:
    lines: list[str] = []
    for k, v in data.items():
        if isinstance(v, dict):
            continue
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, bool):
            lines.append(f'{k} = {"true" if v else "false"}')
        elif isinstance(v, list):
            lines.append(f"{k} = {json.dumps(v)}")
        else:
            lines.append(f"{k} = {v}")
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        section = f"{prefix}.{k}" if prefix else k
        lines.append(f"\n[{section}]")
        lines.append(_simple_toml(v, section))
    return "\n".join(lines) + "\n"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY required (alpha tier).")
        sys.exit(1)

    backend_url = os.getenv("BACKEND_BASE_URL")  # optional

    # Ensure repo root on path so `research.*` can be imported by task app
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    port = int(os.getenv("XBOW_TASK_APP_PORT") or _pick_free_port())
    task_app_cmd = [
        sys.executable,
        "-m",
        "research.customers.xbow.task_app.xbow_task_app",
        "--port",
        str(port),
    ]

    print("=" * 80)
    print("Starting XBOW task app for Context Learning...")
    print("=" * 80)
    print(" ".join(task_app_cmd))

    proc = subprocess.Popen(
        task_app_cmd,
        cwd=str(repo_root),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        health_url = f"http://localhost:{port}/health"
        _wait_for_health(health_url)
        task_app_url = f"http://localhost:{port}"
        print(f"✓ XBOW task app healthy at {task_app_url}")

        from synth_ai.sdk.api.train import ContextLearningJob

        cfg = {
            "context_learning": {
                "task_app_url": task_app_url,
                "evaluation_seeds": [0, 1],
                "environment": {
                    "preflight_script": "echo 'baseline preflight for xbow ctf';\n",
                    "postflight_script": "echo 'baseline postflight';\n",
                },
                "algorithm": {
                    "initial_population_size": 2,
                    "num_generations": 1,
                },
                "metadata": {
                    "experiment_name": "xbow_ctf_context_learning_alpha_cookbook",
                },
            }
        }

        tmp_cfg_path = Path("/tmp/xbow_context_learning_alpha.toml")
        tmp_cfg_path.write_text(_simple_toml(cfg), encoding="utf-8")

        print("\nSubmitting Context Learning job (alpha)...")
        job = ContextLearningJob.from_config(
            tmp_cfg_path,
            backend_url=backend_url,
            api_key=api_key,
        )
        submit = job.submit()
        print(f"  Job ID: {submit.job_id}")

        print("\nStreaming SSE until completion...")
        final = job.stream_until_complete()
        print("\nFinal status:")
        print(final)

        if isinstance(final, dict) and final.get("status") in {"completed", "succeeded"}:
            best = job.download_best_script()
            print("\nBest preflight script:")
            print(best.preflight_script)

    finally:
        print("\nStopping XBOW task app...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()

