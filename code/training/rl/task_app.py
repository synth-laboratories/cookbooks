"""Heart Disease Classification Task App.

This task app evaluates model predictions on the heart disease dataset.
It provides rewards based on classification accuracy.

The task app:
1. Loads the buio/heart-disease dataset from HuggingFace
2. Generates classification tasks from patient data
3. Evaluates predictions and returns binary rewards

Usage:
    python task_app.py                    # Run on default port 8114
    python task_app.py --port 8001        # Run on custom port

Or use the built-in task app:
    python -m synth_ai.sdk.task.server --app heartdisease --port 8114

Requires:
    - ENVIRONMENT_API_KEY environment variable
    - HuggingFace datasets library: pip install datasets
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Feature descriptions for heart disease dataset
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (0 = Female, 1 = Male)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (0/1)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (0/1)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (fixed/normal/reversible)",
}


def load_dataset():
    """Load heart disease dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("buio/heart-disease", split="train")
        return list(dataset)
    except ImportError:
        print("Error: datasets library not installed")
        print("Install with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def format_patient_data(row: dict[str, Any]) -> str:
    """Format patient data as readable text."""
    lines = []

    feature_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    for feature in feature_order:
        if feature in row:
            value = row[feature]
            # Add human-readable descriptions for categorical values
            if feature == "sex":
                value = f"{value} ({'Male' if value == 1 else 'Female'})"
            elif feature == "cp":
                cp_types = ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
                if 0 <= value < len(cp_types):
                    value = f"{value} ({cp_types[value]})"
            elif feature == "fbs":
                value = f"{value} ({'Yes' if value == 1 else 'No'})"
            elif feature == "exang":
                value = f"{value} ({'Yes' if value == 1 else 'No'})"

            lines.append(f"{feature}: {value}")

    return "\n".join(lines)


def create_task_app():
    """Create the heart disease task app."""
    from synth_ai.sdk.task import TaskApp

    # Load dataset
    print("Loading heart disease dataset...")
    data = load_dataset()
    print(f"Loaded {len(data)} patient records")

    # Create task app
    app = TaskApp(
        task_app_id="heartdisease",
        description="Heart disease classification from patient data",
    )

    # Define classification tool
    tool_spec = {
        "type": "function",
        "function": {
            "name": "heart_disease_classify",
            "description": "Submit your classification prediction for whether the patient has heart disease.",
            "parameters": {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "description": "'1' if you predict heart disease, '0' if you predict no heart disease",
                        "enum": ["0", "1"],
                    }
                },
                "required": ["classification"],
            },
        },
    }

    @app.tool("heart_disease_classify")
    def evaluate_classification(classification: str, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the classification prediction."""
        expected = context.get("expected_label")

        if expected is None:
            return {"error": "Missing expected label", "reward": 0.0}

        # Compare prediction to ground truth
        correct = str(classification) == str(expected)

        return {
            "reward": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted": classification,
            "expected": str(expected),
        }

    @app.task_generator
    def generate_task(index: int) -> dict[str, Any] | None:
        """Generate a classification task from the dataset."""
        if index >= len(data):
            return None

        row = data[index]
        patient_data = format_patient_data(row)

        # Get label (handle different column names)
        label = row.get("target", row.get("label", row.get("num", 0)))
        # Binarize if needed (some versions have 0-4 scale)
        if isinstance(label, int) and label > 1:
            label = 1

        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a medical classifier. Analyze the patient data and "
                        "predict whether they have heart disease. Use the heart_disease_classify "
                        "tool to submit your prediction: '1' for heart disease, '0' for no disease."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Please classify this patient:\n\n{patient_data}",
                },
            ],
            "tools": [tool_spec],
            "context": {
                "expected_label": label,
                "patient_index": index,
            },
        }

    return app


def main():
    """Run the task app server."""
    parser = argparse.ArgumentParser(description="Heart Disease Task App")
    parser.add_argument("--port", type=int, default=8114, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    # Check environment key
    env_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not env_key:
        print("Warning: ENVIRONMENT_API_KEY not set")
        print("Task app will run but may not authenticate with backend")

    # Create and run app
    app = create_task_app()

    print(f"\nStarting task app on http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  Health: http://localhost:{args.port}/health")
    print(f"  Task Info: http://localhost:{args.port}/task_info")
    print("\nPress Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
