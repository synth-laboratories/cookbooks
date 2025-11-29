"""In-Process Task App Example for RL.

This script demonstrates running a task app in the same process:
1. Define a task app using the SDK decorators
2. Use InProcessTaskApp context manager
3. Submit RL job while task app runs locally

This is useful for:
- Local development and testing
- Custom reward functions
- Rapid iteration without deploying task apps

Usage:
    python in_process.py

Requires:
    - SYNTH_API_KEY environment variable
    - HuggingFace datasets library
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def create_heart_disease_task_app():
    """Create heart disease task app definition."""
    from synth_ai.sdk.task import TaskApp, tool

    # Load dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset("buio/heart-disease", split="train")
        data_list = list(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Install datasets: pip install datasets")
        sys.exit(1)

    # Create task app
    app = TaskApp(
        task_app_id="heartdisease-inprocess",
        description="Heart disease classification task (in-process)",
    )

    # Define the classification tool
    @tool(
        name="heart_disease_classify",
        description="Submit your classification prediction for heart disease.",
        parameters={
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "description": "'1' for heart disease, '0' for no disease",
                    "enum": ["0", "1"],
                }
            },
            "required": ["classification"],
        },
    )
    def classify(classification: str, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate classification prediction."""
        expected = context.get("expected_label")

        if expected is None:
            return {"error": "Missing expected label in context"}

        correct = str(classification) == str(expected)
        return {
            "reward": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted": classification,
            "expected": str(expected),
        }

    app.add_tool(classify)

    # Define task generator
    @app.task_generator
    def generate_task(index: int) -> dict[str, Any]:
        """Generate a task from the dataset."""
        if index >= len(data_list):
            return None  # No more tasks

        row = data_list[index]

        # Format patient data as prompt
        features = []
        feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        for name in feature_names:
            if name in row:
                features.append(f"{name}: {row[name]}")

        patient_data = "\n".join(features)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical classifier. Based on patient data, predict if they have heart disease. Use the heart_disease_classify tool with '1' for disease or '0' for no disease.",
                },
                {
                    "role": "user",
                    "content": f"Patient Features:\n{patient_data}\n\nClassify this patient.",
                },
            ],
            "tools": [app.get_tool_spec("heart_disease_classify")],
            "context": {
                "expected_label": row.get("target", row.get("label", 0)),
                "index": index,
            },
        }

    return app


def main() -> None:
    """Run RL training with in-process task app."""
    from synth_ai.sdk.api.train.rl import RLJob
    from synth_ai.sdk.task import InProcessTaskApp

    # Verify environment
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY not set")
        sys.exit(1)

    print("Creating in-process task app...")
    task_app = create_heart_disease_task_app()

    # Config path
    config_path = Path(__file__).parent.parent / "configs" / "rl.toml"

    print("Starting in-process task app and submitting job...")
    print("=" * 50)

    # Use InProcessTaskApp context manager
    # This starts the task app locally and provides a URL
    with InProcessTaskApp(task_app, port=0) as local_app:
        print(f"Task app running at: {local_app.url}")

        # Create job with local task app URL
        job = RLJob.from_config(
            config_path=config_path,
            task_url_override=local_app.url,
        )

        try:
            job_id = job.submit()
            print(f"Job submitted: {job_id}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        print("\nPolling for completion...")
        print("(Task app is running in this process)")
        print("-" * 50)

        # Poll until complete
        result = job.poll_until_complete(
            timeout=3600.0,
            interval=10.0,
        )

        print("-" * 50)

    # Task app automatically stopped when exiting context
    print("\nTask app stopped.")

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    status = result.get("status", "unknown")
    print(f"Status: {status}")

    if status == "completed":
        final_reward = result.get("final_reward", "N/A")
        print(f"Final Reward: {final_reward}")

        # Get metrics
        metrics = job.get_metrics()
        if metrics and "steps" in metrics:
            steps = metrics["steps"]
            print(f"Training Steps: {len(steps)}")

            if steps:
                rewards = [s.get("reward", 0) for s in steps]
                print(f"Reward Range: {min(rewards):.4f} - {max(rewards):.4f}")
    else:
        print(f"Job did not complete successfully: {result}")


if __name__ == "__main__":
    main()
