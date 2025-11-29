"""Banking77 Task App for Prompt Learning.

This task app evaluates prompt candidates on the Banking77 intent classification
dataset from HuggingFace. It's used by both MIPRO and GEPA optimizers.

Dataset: banking77 from HuggingFace
- 10,003 train examples
- 3,080 test examples
- 77 banking intent categories

Output Modes:
    This task app demonstrates TWO output modes:
    1. TOOL_CALLS (default): Uses function calling with tool_choice
    2. STRUCTURED: Uses response_format with json_schema for simpler extraction

    Set OUTPUT_MODE=structured in environment to use structured outputs.

Usage:
    python task_app.py

Environment:
    ENVIRONMENT_API_KEY: API key for authenticating optimizer requests
    OPENAI_API_KEY: API key for making LLM calls (or GROQ_API_KEY)
    PORT: Server port (default: 8001)
    OUTPUT_MODE: "tool_calls" (default) or "structured"

Requires:
    pip install fastapi uvicorn datasets httpx
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Banking77 Task App")

# Configuration
ENVIRONMENT_API_KEY = os.getenv("ENVIRONMENT_API_KEY", "dev-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PORT = int(os.getenv("PORT", "8001"))
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "tool_calls").lower()  # "tool_calls" or "structured"

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

# Dataset will be loaded at startup
DATASET: list[dict[str, str]] = []


def load_banking77_dataset() -> list[dict[str, str]]:
    """Load Banking77 dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading Banking77 from HuggingFace...")
        dataset = load_dataset("banking77", split="test")  # Use test split for evaluation
        return [
            {
                "query": example["text"],
                "label": BANKING77_LABELS[example["label"]] if example["label"] < len(BANKING77_LABELS) else str(example["label"]),
            }
            for example in dataset
        ]
    except ImportError:
        print("Warning: datasets library not installed. Using fallback data.")
        print("Install with: pip install datasets")
        return [
            {"query": "I want to cancel my card", "label": "card_about_to_expire"},
            {"query": "How do I transfer money?", "label": "transfer"},
            {"query": "What's my balance?", "label": "balance_not_updated_after_bank_transfer"},
        ]
    except Exception as e:
        print(f"Warning: Failed to load dataset: {e}. Using fallback data.")
        return [
            {"query": "I want to cancel my card", "label": "card_about_to_expire"},
            {"query": "How do I transfer money?", "label": "transfer"},
            {"query": "What's my balance?", "label": "balance_not_updated_after_bank_transfer"},
        ]


def verify_api_key(x_api_key: str | None) -> bool:
    """Verify the X-API-Key header matches ENVIRONMENT_API_KEY."""
    return x_api_key == ENVIRONMENT_API_KEY


def call_llm(prompt: str, query: str, inference_url: str | None = None) -> str:
    """Call the LLM with the given prompt and query.

    Supports two output modes (controlled by OUTPUT_MODE env var):
    - tool_calls: Uses function calling with tool_choice (default)
    - structured: Uses response_format with json_schema (simpler)
    """
    import httpx

    # Determine base URL and API key
    if inference_url:
        # Handle query params in inference_url
        if "?" in inference_url:
            base, query_string = inference_url.split("?", 1)
            url = f"{base.rstrip('/')}/chat/completions?{query_string}"
        else:
            url = f"{inference_url.rstrip('/')}/chat/completions"

        # Use appropriate API key based on URL
        if "groq" in url.lower():
            api_key = GROQ_API_KEY
        else:
            api_key = OPENAI_API_KEY
    else:
        url = "https://api.openai.com/v1/chat/completions"
        api_key = OPENAI_API_KEY

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build payload based on output mode
    payload: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    }

    if OUTPUT_MODE == "structured":
        # STRUCTURED OUTPUT MODE: Uses response_format with json_schema
        # This is simpler and more direct for classification tasks
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "The classified banking intent",
                        }
                    },
                    "required": ["intent"],
                    "additionalProperties": False,
                },
            },
        }
    else:
        # TOOL_CALLS MODE (default): Uses function calling with forced tool_choice
        # This is the traditional approach and works with more models
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "classify",
                    "description": "Classify the user's banking intent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "description": "The classified intent",
                            }
                        },
                        "required": ["intent"],
                    },
                },
            }
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": "classify"}}

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    # Extract intent based on output mode
    if OUTPUT_MODE == "structured":
        # STRUCTURED MODE: Parse JSON from message.content
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("intent", "unknown")
        except (KeyError, IndexError, json.JSONDecodeError):
            return "unknown"
    else:
        # TOOL_CALLS MODE: Extract from tool_calls[].function.arguments
        try:
            tool_call = data["choices"][0]["message"]["tool_calls"][0]
            args = json.loads(tool_call["function"]["arguments"])
            return args.get("intent", "unknown")
        except (KeyError, IndexError, json.JSONDecodeError):
            return "unknown"


@app.get("/health")
async def health():
    """Health check endpoint (unauthenticated)."""
    return {"status": "ok"}


@app.get("/task_info")
async def task_info(x_api_key: str | None = Header(None)):
    """Return dataset metadata (authenticated)."""
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return {
        "task_app_id": "banking77_intent_classification",
        "dataset_size": len(DATASET),
        "labels": list(set(d["label"] for d in DATASET)),
    }


@app.post("/rollout")
async def rollout(request: Request, x_api_key: str | None = Header(None)):
    """Evaluate a prompt candidate on the dataset (authenticated)."""
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Extract prompt and config
    env = data.get("env", {})
    policy = data.get("policy", {})
    config = policy.get("config", {})

    seed = env.get("seed", 0)
    inference_url = config.get("inference_url")
    prompt_template = config.get("prompt_template", "Classify the user's banking intent.")

    # Get sample from dataset
    sample_idx = seed % len(DATASET)
    sample = DATASET[sample_idx]
    query = sample["query"]
    true_label = sample["label"]

    # Call LLM with prompt
    try:
        predicted_intent = call_llm(prompt_template, query, inference_url)
    except Exception as e:
        return JSONResponse({
            "error": f"LLM call failed: {e}",
            "metrics": {"mean_return": 0.0},
            "trajectories": [{
                "steps": [{"reward": 0.0}],
            }],
        })

    # Compute reward (1.0 if correct, 0.0 otherwise)
    reward = 1.0 if predicted_intent.lower() == true_label.lower() else 0.0

    return {
        "metrics": {"mean_return": reward},
        "trajectories": [
            {
                "steps": [
                    {
                        "observation": {"query": query, "true_label": true_label},
                        "action": {"predicted_intent": predicted_intent},
                        "reward": reward,
                    }
                ],
            }
        ],
    }


@app.on_event("startup")
async def startup_event():
    """Load dataset on startup."""
    global DATASET
    DATASET = load_banking77_dataset()
    print(f"Dataset loaded: {len(DATASET)} examples")


if __name__ == "__main__":
    print(f"Starting Banking77 Task App on port {PORT}")
    print(f"Output mode: {OUTPUT_MODE}")
    print(f"Health check: http://localhost:{PORT}/health")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
