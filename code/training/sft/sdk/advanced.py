"""Advanced SFT SDK Example.

This script demonstrates advanced SDK features:
1. Download and validate Banking77 from HuggingFace
2. Progress monitoring with callbacks
3. Checkpoint handling
4. Evaluation comparison (base vs fine-tuned)

Usage:
    python advanced.py [--mode validate|train|evaluate]

Requires:
    - SYNTH_API_KEY environment variable
    - HuggingFace datasets library: pip install datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Banking77 intent labels
BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement",
    "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
    "top_up_limits", "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def prepare_banking77_splits() -> tuple[Path, Path]:
    """Download Banking77 and prepare train/test splits."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed")
        print("Install with: pip install datasets")
        sys.exit(1)

    print("Downloading Banking77 dataset from HuggingFace...")
    train_ds = load_dataset("banking77", split="train")
    test_ds = load_dataset("banking77", split="test")

    temp_dir = tempfile.mkdtemp()
    train_path = Path(temp_dir) / "train.jsonl"
    test_path = Path(temp_dir) / "test.jsonl"

    def write_split(dataset, path):
        with open(path, "w") as f:
            for example in dataset:
                label = BANKING77_LABELS[example["label"]] if example["label"] < len(BANKING77_LABELS) else str(example["label"])
                messages = {
                    "messages": [
                        {"role": "system", "content": "You are a banking assistant that classifies customer intents."},
                        {"role": "user", "content": example["text"]},
                        {"role": "assistant", "content": label},
                    ]
                }
                f.write(json.dumps(messages) + "\n")

    print(f"Converting {len(train_ds)} train examples...")
    write_split(train_ds, train_path)

    print(f"Converting {len(test_ds)} test examples...")
    write_split(test_ds, test_path)

    return train_path, test_path


def validate_dataset(dataset_path: Path) -> bool:
    """Validate dataset format before training."""
    from synth_ai.sdk.learning.sft.data import validate_jsonl_or_raise

    print(f"Validating dataset: {dataset_path}")

    try:
        validate_jsonl_or_raise(
            dataset_path,
            min_messages=2,  # At least user + assistant
        )
        print("Dataset validation passed!")
        return True
    except ValueError as e:
        print(f"Dataset validation failed: {e}")
        return False


def on_progress(status: dict[str, Any]) -> None:
    """Callback for training progress updates."""
    current_status = status.get("status", "unknown")
    progress = status.get("progress", {})

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Status: {current_status}", end="")

    if "step" in progress:
        total = progress.get("total_steps", "?")
        print(f" | Step: {progress['step']}/{total}", end="")
    if "loss" in progress:
        print(f" | Loss: {progress['loss']:.4f}", end="")
    if "learning_rate" in progress:
        print(f" | LR: {progress['learning_rate']:.2e}", end="")

    print()


def train_with_monitoring(config_path: Path, dataset_path: Path | None = None) -> dict[str, Any]:
    """Run training with detailed progress monitoring."""
    from synth_ai.sdk.api.train.sft import SFTJob

    print(f"\n{'='*60}")
    print("Starting SFT Training with Monitoring")
    print(f"{'='*60}")

    job = SFTJob.from_config(
        config_path=config_path,
        dataset_path=dataset_path,
    )

    print("Submitting job...")

    try:
        job_id = job.submit()
        print(f"Job ID: {job_id}")
    except Exception as e:
        print(f"Submission failed: {e}")
        return {"status": "failed", "error": str(e)}

    print("Training started. Monitoring progress...")

    # Poll with progress callback
    try:
        result = job.poll_until_complete(
            timeout=7200.0,
            interval=30.0,  # Check every 30 seconds
            on_status=on_progress,
        )
    except TimeoutError:
        print("Training timed out!")
        return {"status": "timeout", "job_id": job_id}

    if result.get("status") == "completed":
        model_id = job.get_fine_tuned_model()
        return {
            "status": "completed",
            "job_id": job_id,
            "model_id": model_id,
            "final_loss": result.get("final_loss"),
        }
    else:
        return {
            "status": result.get("status", "unknown"),
            "job_id": job_id,
            "error": result.get("error"),
        }


def evaluate_models(base_model: str, fine_tuned_model: str, test_data_path: Path) -> None:
    """Compare base model vs fine-tuned model on test data."""
    import asyncio
    from synth_ai.sdk.inference import InferenceClient

    print(f"\n{'='*60}")
    print("Model Evaluation Comparison")
    print(f"{'='*60}")
    print(f"Base model: {base_model}")
    print(f"Fine-tuned: {fine_tuned_model}")

    # Load test data
    with open(test_data_path) as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    print(f"Test samples: {len(test_data)}")

    client = InferenceClient(
        base_url="https://agent-learning.onrender.com",
        api_key=os.environ["SYNTH_API_KEY"],
    )

    async def evaluate_sample(model: str, messages: list[dict]) -> str:
        """Get model response for a sample."""
        try:
            response = await client.create_chat_completion(
                model=model,
                messages=messages[:-1],  # Exclude the assistant turn
                max_tokens=200,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"ERROR: {e}"

    async def run_evaluation():
        base_correct = 0
        ft_correct = 0

        for i, sample in enumerate(test_data[:10]):  # Evaluate first 10
            messages = sample.get("messages", [])
            if len(messages) < 2:
                continue

            expected = messages[-1]["content"]

            base_resp = await evaluate_sample(base_model, messages)
            ft_resp = await evaluate_sample(fine_tuned_model, messages)

            # Simple exact match (you'd want better metrics in practice)
            base_match = expected.lower().strip() in base_resp.lower()
            ft_match = expected.lower().strip() in ft_resp.lower()

            if base_match:
                base_correct += 1
            if ft_match:
                ft_correct += 1

            print(f"\n[Sample {i+1}]")
            print(f"  Expected: {expected[:50]}...")
            print(f"  Base: {base_resp[:50]}... ({'MATCH' if base_match else 'NO MATCH'})")
            print(f"  FT:   {ft_resp[:50]}... ({'MATCH' if ft_match else 'NO MATCH'})")

        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  Base model matches: {base_correct}/10")
        print(f"  Fine-tuned matches: {ft_correct}/10")
        print(f"  Improvement: {((ft_correct - base_correct) / max(base_correct, 1)) * 100:+.1f}%")

    asyncio.run(run_evaluation())


def main() -> None:
    """Run advanced examples."""
    parser = argparse.ArgumentParser(description="Advanced SFT SDK Examples")
    parser.add_argument(
        "--mode",
        choices=["validate", "train", "evaluate"],
        default="train",
        help="Mode: validate dataset, train model, or evaluate results",
    )
    parser.add_argument(
        "--model-id",
        help="Fine-tuned model ID for evaluation mode",
    )

    args = parser.parse_args()

    # Paths
    cookbook_dir = Path(__file__).parent.parent
    config_path = cookbook_dir / "configs" / "sft.toml"

    # Verify environment
    if args.mode != "validate" and not os.environ.get("SYNTH_API_KEY"):
        print("Error: SYNTH_API_KEY not set")
        sys.exit(1)

    if args.mode == "validate":
        # Download and validate Banking77
        train_path, _ = prepare_banking77_splits()
        valid = validate_dataset(train_path)
        sys.exit(0 if valid else 1)

    elif args.mode == "train":
        if not config_path.exists():
            print(f"Error: Config not found: {config_path}")
            sys.exit(1)

        # Prepare Banking77 dataset
        train_path, test_path = prepare_banking77_splits()

        result = train_with_monitoring(config_path, train_path)

        # Save results
        output_path = cookbook_dir / "results" / "training_result.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "evaluate":
        if not args.model_id:
            print("Error: --model-id required for evaluate mode")
            sys.exit(1)

        # Prepare test data from Banking77
        _, test_path = prepare_banking77_splits()

        evaluate_models(
            base_model="Qwen/Qwen3-0.6B",
            fine_tuned_model=args.model_id,
            test_data_path=test_path,
        )


if __name__ == "__main__":
    main()
