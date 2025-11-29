"""Basic SFT SDK Example.

This script demonstrates the simplest SDK workflow for supervised fine-tuning:
1. Download Banking77 dataset from HuggingFace
2. Convert to JSONL format for training
3. Load config from TOML file
4. Submit job (handles file upload automatically)
5. Poll until complete
6. Get fine-tuned model ID

Usage:
    python basic.py

Requires:
    - SYNTH_API_KEY environment variable
    - HuggingFace datasets library: pip install datasets
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


# Banking77 intent labels (77 categories)
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


def prepare_banking77_dataset() -> Path:
    """Download and prepare Banking77 dataset for SFT training."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed")
        print("Install with: pip install datasets")
        sys.exit(1)

    print("Downloading Banking77 dataset from HuggingFace...")
    dataset = load_dataset("banking77", split="train")

    print(f"Converting {len(dataset)} examples to JSONL format...")

    # Create temp file for training data
    temp_dir = tempfile.mkdtemp()
    train_path = Path(temp_dir) / "banking77_train.jsonl"

    with open(train_path, "w") as f:
        for example in dataset:
            text = example["text"]
            label_idx = example["label"]
            label = BANKING77_LABELS[label_idx] if label_idx < len(BANKING77_LABELS) else str(label_idx)

            # Format as chat messages
            messages = {
                "messages": [
                    {"role": "system", "content": "You are a banking assistant that classifies customer intents into one of 77 categories."},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": label},
                ]
            }
            f.write(json.dumps(messages) + "\n")

    print(f"Training data saved to: {train_path}")
    return train_path


def main() -> None:
    """Run basic SFT workflow."""
    from synth_ai.sdk.api.train.sft import SFTJob

    # Verify environment
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY not set")
        print("Set it with: export SYNTH_API_KEY=your-key")
        sys.exit(1)

    # Config path
    config_path = Path(__file__).parent.parent / "configs" / "sft.toml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    # Prepare dataset from HuggingFace
    dataset_path = prepare_banking77_dataset()

    print(f"\nLoading config from: {config_path}")

    # Create job from config
    job = SFTJob.from_config(
        config_path=config_path,
        dataset_path=dataset_path,
    )

    print("Submitting job (uploading dataset)...")

    # Submit job
    try:
        job_id = job.submit()
        print(f"Job submitted: {job_id}")
    except Exception as e:
        print(f"Error: Job submission failed: {e}")
        sys.exit(1)

    print("Polling for completion (this may take 10-30 minutes)...")

    # Poll until complete
    result = job.poll_until_complete(timeout=7200.0)

    print("\n=== Results ===")
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get("status") == "completed":
        # Get fine-tuned model ID
        model_id = job.get_fine_tuned_model()
        print(f"Fine-tuned model: {model_id}")

        # Save for later use
        output_path = Path(__file__).parent.parent / "results" / "model_info.txt"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"job_id: {job_id}\n")
            f.write(f"model_id: {model_id}\n")
        print(f"\nModel info saved to: {output_path}")
    else:
        print(f"Job did not complete successfully: {result}")


if __name__ == "__main__":
    main()
