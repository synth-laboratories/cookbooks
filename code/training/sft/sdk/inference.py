"""Using Fine-Tuned Models for Inference.

This script demonstrates using fine-tuned models:
1. Load model ID from training results
2. Make inference requests
3. Compare with base model

Usage:
    python inference.py [model_id]

Requires:
    - SYNTH_API_KEY environment variable
    - Fine-tuned model ID (from training)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


async def main() -> None:
    """Demonstrate inference with fine-tuned model."""
    from synth_ai.sdk.inference import InferenceClient

    # Verify environment
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY not set")
        sys.exit(1)

    # Get model ID from args or results file
    model_id = None
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        # Try to load from results
        results_path = Path(__file__).parent.parent / "results" / "training_result.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            model_id = results.get("model_id")

    if not model_id:
        print("Usage: python inference.py <model_id>")
        print("Or run training first to create results/training_result.json")
        sys.exit(1)

    print(f"Using fine-tuned model: {model_id}")

    # Create inference client
    client = InferenceClient(
        base_url="https://agent-learning.onrender.com",
        api_key=api_key,
    )

    # Example messages
    test_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to add two numbers."},
        ],
        [
            {"role": "user", "content": "Explain quantum computing in one sentence."},
        ],
    ]

    print("\n" + "="*60)
    print("Inference Examples")
    print("="*60)

    for i, messages in enumerate(test_messages, 1):
        print(f"\n[Example {i}]")
        print(f"User: {messages[-1]['content']}")

        try:
            response = await client.create_chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )

            assistant_content = response["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_content}")

        except Exception as e:
            print(f"Error: {e}")

    # Streaming example
    print("\n" + "="*60)
    print("Streaming Example")
    print("="*60)

    stream_messages = [
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a very short story about a robot."},
    ]

    print("User:", stream_messages[-1]["content"])
    print("Assistant: ", end="", flush=True)

    try:
        async for chunk in client.create_chat_completion_stream(
            model=model_id,
            messages=stream_messages,
            max_tokens=150,
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            print(content, end="", flush=True)
        print()  # Newline at end
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
