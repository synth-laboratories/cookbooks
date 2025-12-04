#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "synth-ai>=0.2.26",
#     "python-dotenv>=1.0.0",
#     "httpx>=0.24.0",
#     "click>=8.1.0",
# ]
# ///
"""
Comprehensive Banking77 GEPA Demo

This script runs a comprehensive GEPA optimization on Banking77 and saves all necessary
data for in-depth analysis:

- Scores for each candidate (train and validation)
- Prompts for each candidate
- Per-seed scores for each candidate
- Confusion matrices
- Pareto frontier
- Optimistic scores
- Cost and time data

Results are saved to results/ directory for analysis.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Disable verbose task app logging
if "TASK_APP_VERBOSE" not in os.environ:
    os.environ["TASK_APP_VERBOSE"] = "0"

# Disable uvicorn access logs
import logging

try:
    import uvicorn

    original_config_init = uvicorn.Config.__init__

    def patched_config_init(self, *args, **kwargs):
        kwargs.setdefault("access_log", False)
        return original_config_init(self, *args, **kwargs)

    uvicorn.Config.__init__ = patched_config_init
except Exception:
    pass

logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)

# Load env vars from .env file in current directory or parent directories
load_dotenv(override=False)

from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.task.in_process import InProcessTaskApp
from synth_ai.core.urls import BACKEND_URL_BASE


class Banking77DemoTracker:
    """Tracks GEPA optimization progress and collects all data for analysis."""

    def __init__(self):
        self.start_time = time.time()
        self.candidates: list[dict[str, Any]] = []
        self.baseline_score: float | None = None
        self.baseline_prompt: dict[str, Any] | None = None
        self.baseline_instance_scores: list[float] = []
        self.best_score = 0.0
        self.pareto_frontier: list[dict[str, Any]] = []
        self.pareto_history: list[dict[str, Any]] = []  # Track frontier updates over time
        self.usage_data: dict[str, Any] | None = None
        self.finish_reason: str | None = None
        self.status = "starting"

    def update_from_event(self, event: dict[str, Any]) -> None:
        """Update tracker from SSE event."""
        event_type = event.get("type", "").replace("[MASKED]", "gepa")
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

        if "baseline" in event_type.lower():
            self.baseline_score = data.get("baseline_score") or data.get("accuracy")

        elif (
            "candidate.evaluated" in event_type.lower()
            or "candidate_scored" in event_type.lower()
            or "proposal.scored" in event_type.lower()
            or "optimized.scored" in event_type.lower()
        ):
            try:
                prompt_text = self._extract_prompt_text(data) if data else None
            except (RecursionError, Exception):
                # Fallback to simple extraction if recursion or other error
                prompt_text = data.get("prompt_text") if data else None
                if not prompt_text and data:
                    prompt_text = (data.get("object", {}) or {}).get("prompt_text")
            
            candidate = {
                "candidate_id": data.get("candidate_id") or data.get("version_id", f"c{len(self.candidates)}"),
                "accuracy": data.get("accuracy") or data.get("score", 0),
                "val_accuracy": data.get("val_accuracy"),
                "train_accuracy": data.get("train_accuracy"),
                "prompt_text": prompt_text,
                "generation": data.get("generation"),
                "parent_id": data.get("parent_id"),
                "is_pareto": data.get("is_pareto", False),
                "timestamp": time.time(),
                "raw_data": data,
            }
            self.candidates.append(candidate)
            if candidate["accuracy"] and candidate["accuracy"] > self.best_score:
                self.best_score = candidate["accuracy"]

        elif "frontier_updated" in event_type.lower():
            frontier = data.get("frontier", [])
            added = data.get("added", [])
            removed = data.get("removed", [])
            frontier_size = data.get("frontier_size", len(frontier))
            self.pareto_frontier = [{"candidate_id": fid, "score": data.get("best_score", 0)} for fid in frontier]
            # Track pareto history
            self.pareto_history.append({
                "timestamp": time.time() - self.start_time,
                "added": added,
                "removed": removed,
                "frontier_size": frontier_size,
                "frontier": frontier,
            })

        elif (
            event_type == "prompt.learning.completed"
            or event_type == "prompt.learning.gepa.complete"
            or (event_type.endswith(".complete") and "generation" not in event_type.lower() and "proposal" not in event_type.lower())
        ):
            self.status = "complete"
            self.best_score = data.get("best_score", self.best_score)
            self.finish_reason = data.get("finish_reason")
        
        elif event_type in ("prompt.learning.failed", "prompt.learning.error"):
            self.status = "failed"
            self.finish_reason = data.get("error") or data.get("error_message") or "Job failed"

        elif "usage.recorded" in event_type.lower():
            self.usage_data = data

    def _extract_prompt_text(self, data: dict[str, Any]) -> str | None:
        """Extract prompt text from various data structures."""
        try:
            return self._extract_prompt_text_from_candidate(data)
        except RecursionError:
            # Fallback to simple extraction
            return data.get("prompt_text") or (data.get("object", {}) or {}).get("prompt_text")
    
    def _extract_prompt_text_from_candidate(self, cand: dict[str, Any], depth: int = 0, _seen: set | None = None) -> str | None:
        """Extract prompt text from candidate data."""
        # Prevent infinite recursion with strict depth limit and identity check
        if depth > 2:
            return None

        # Track seen objects to prevent cycles
        if _seen is None:
            _seen = set()
        cand_id = id(cand)
        if cand_id in _seen:
            return None
        _seen.add(cand_id)

        # Try direct prompt_text (must be a string)
        prompt_text = cand.get("prompt_text")
        if prompt_text and isinstance(prompt_text, str):
            return prompt_text

        # Try object.prompt_text
        obj = cand.get("object")
        if isinstance(obj, dict):
            if obj.get("prompt_text") and isinstance(obj.get("prompt_text"), str):
                return obj["prompt_text"]
            # Try prompt_sections
            if obj.get("prompt_sections"):
                sections = obj["prompt_sections"]
                parts = []
                for s in sections:
                    if isinstance(s, dict):
                        role = s.get("role", "unknown")
                        content = s.get("content", "")
                        if content and isinstance(content, str):
                            parts.append(f"[{role.upper()}]: {content}")
                if parts:
                    return "\n\n".join(parts)

        # Try messages format
        messages = cand.get("messages", [])
        if messages and isinstance(messages, list):
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content and isinstance(content, str):
                        parts.append(f"[{role.upper()}]: {content}")
            if parts:
                return "\n\n".join(parts)

        # Try prompt_sections at top level
        if cand.get("prompt_sections"):
            sections = cand["prompt_sections"]
            parts = []
            for s in sections:
                if isinstance(s, dict):
                    role = s.get("role", "unknown")
                    content = s.get("content", "")
                    if content and isinstance(content, str):
                        parts.append(f"[{role.upper()}]: {content}")
            if parts:
                return "\n\n".join(parts)

        # Try sections format
        if cand.get("sections"):
            sections = cand["sections"]
            parts = []
            for s in sections:
                if isinstance(s, dict):
                    role = s.get("role", "unknown")
                    content = s.get("content", "") or s.get("pattern", "")
                    if content and isinstance(content, str):
                        parts.append(f"[{role.upper()}]: {content}")
            if parts:
                return "\n\n".join(parts)

        # Try best_prompt format - NO RECURSION, just extract directly
        best_prompt = cand.get("best_prompt")
        if best_prompt and isinstance(best_prompt, dict):
            # Check if best_prompt has prompt_text directly
            bp_text = best_prompt.get("prompt_text")
            if bp_text and isinstance(bp_text, str):
                return bp_text
            # Check if best_prompt has messages
            bp_messages = best_prompt.get("messages")
            if bp_messages and isinstance(bp_messages, list):
                parts = []
                for msg in bp_messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if content and isinstance(content, str):
                            parts.append(f"[{role.upper()}]: {content}")
                if parts:
                    return "\n\n".join(parts)

        return None


def build_confusion_matrix_from_trajectories(
    trajectories: list[dict[str, Any]], label_names: list[str]
) -> dict[str, Any]:
    """Build confusion matrix from rollout trajectories."""
    from collections import defaultdict

    # Initialize confusion matrix counts
    confusion_counts: dict[tuple[str, str], int] = defaultdict(int)
    all_intents = set(label_names)

    # Extract predictions and ground truth from trajectories
    for traj in trajectories:
        steps = traj.get("steps", [])
        for step in steps:
            info = step.get("info", {})
            expected_intent = info.get("expected_intent", "")
            predicted_intent = info.get("predicted_intent", "")

            if expected_intent and predicted_intent:
                # Normalize intent names (handle underscores/spaces)
                expected_norm = expected_intent.lower().replace("_", " ").strip()
                predicted_norm = predicted_intent.lower().replace("_", " ").strip()
                confusion_counts[(expected_norm, predicted_norm)] += 1
                all_intents.add(expected_norm)
                all_intents.add(predicted_norm)

    # Build matrix as list of lists for easier JSON serialization
    sorted_intents = sorted(all_intents)
    intent_to_idx = {intent: i for i, intent in enumerate(sorted_intents)}
    matrix = [[0 for _ in sorted_intents] for _ in sorted_intents]

    for (expected, predicted), count in confusion_counts.items():
        if expected in intent_to_idx and predicted in intent_to_idx:
            matrix[intent_to_idx[expected]][intent_to_idx[predicted]] = count

    return {
        "intents": sorted_intents,
        "matrix": matrix,
        "counts": {f"{exp} -> {pred}": count for (exp, pred), count in confusion_counts.items()},
    }


def save_comprehensive_results(
    output_dir: Path,
    tracker: Banking77DemoTracker,
    job_results: dict[str, Any],
    prompt_results: Any,
    scoring_summary: dict[str, Any],
) -> None:
    """Save all comprehensive results for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper to safely extract numeric accuracy
    def _safe_acc(val):
        if val is None:
            return None
        if isinstance(val, dict):
            return float(val.get("score", 0) or val.get("accuracy", 0) or 0)
        return float(val)

    # Extract configuration seeds from job_results
    train_seeds = job_results.get("train_seeds", [])
    val_seeds = job_results.get("val_seeds", [])
    config = job_results.get("config", {})
    if not train_seeds and config:
        gepa_config = config.get("gepa", {})
        eval_config = gepa_config.get("evaluation", {})
        train_seeds = eval_config.get("train_seeds", [])
        val_seeds = eval_config.get("val_seeds", [])

    # 0. Save run configuration
    config_file = output_dir / "run_config.json"
    run_config_data = {
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "num_train_seeds": len(train_seeds),
        "num_val_seeds": len(val_seeds),
        "job_config": config,
    }
    with open(config_file, "w") as f:
        json.dump(run_config_data, f, indent=2)
    print(f"✅ Saved run config to {config_file} (train_seeds: {len(train_seeds)}, val_seeds: {len(val_seeds)})")

    # 1. Save candidates with scores and FULL prompts
    candidates_file = output_dir / "candidates.json"

    # Extract full prompt from raw_data if prompt_text is missing
    def _get_full_prompt(c):
        if c.get("prompt_text"):
            return c["prompt_text"]
        raw = c.get("raw_data", {})
        if raw:
            # Try to extract from raw_data
            return tracker._extract_prompt_text_from_candidate(raw) if hasattr(tracker, '_extract_prompt_text_from_candidate') else None
        return None

    candidates_data = {
        "baseline_score": tracker.baseline_score,
        "baseline_instance_scores": tracker.baseline_instance_scores,
        "best_score": tracker.best_score,
        "total_candidates": len(tracker.candidates),
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "train_accuracy": _safe_acc(c.get("accuracy")),
                "val_accuracy": _safe_acc(c.get("val_accuracy")),
                "generation": c.get("generation"),
                "parent_id": c.get("parent_id"),
                "is_pareto": c.get("is_pareto"),
                "instance_scores": c.get("instance_scores", []),
                "prompt_text": _get_full_prompt(c),
                # Include raw prompt sections if available
                "prompt_sections": c.get("raw_data", {}).get("object", {}).get("prompt_sections") if isinstance(c.get("raw_data"), dict) else None,
                "messages": c.get("raw_data", {}).get("messages") if isinstance(c.get("raw_data"), dict) else None,
            }
            for c in tracker.candidates
        ],
    }
    with open(candidates_file, "w") as f:
        json.dump(candidates_data, f, indent=2)
    print(f"✅ Saved candidates to {candidates_file}")

    # 2. Save per-seed scores for each candidate
    per_seed_file = output_dir / "per_seed_scores.json"
    per_seed_data = {
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "baseline": {},
        "candidates": {},
    }

    # Extract baseline scores from job_results
    baseline_scores = job_results.get("baseline_scores", {})
    baseline_instance_scores = job_results.get("baseline_instance_scores", [])
    if not baseline_instance_scores:
        # Try from scoring_summary
        baseline_instance_scores = scoring_summary.get("baseline_instance_scores", [])

    if baseline_instance_scores:
        per_seed_data["baseline"] = {
            "accuracy": tracker.baseline_score,
            "per_seed_scores": [
                {"seed": train_seeds[i] if i < len(train_seeds) else i, "score": float(score) if score is not None else 0.0}
                for i, score in enumerate(baseline_instance_scores)
            ],
            "num_seeds": len(baseline_instance_scores),
        }
    elif tracker.baseline_score is not None:
        per_seed_data["baseline"] = {
            "accuracy": tracker.baseline_score,
            "per_seed_scores": [],
            "note": "Per-seed baseline scores not available",
        }

    # Extract per-seed scores from validation results (if available)
    validation_results = job_results.get("validation_results", [])

    for val_result in validation_results:
        version_id = val_result.get("version_id") or val_result.get("candidate_id")
        if not version_id:
            rank = val_result.get("rank")
            if rank is not None and rank >= 0:
                matching_cand = next(
                    (c for c in tracker.candidates if c.get("rank") == rank),
                    None
                )
                if matching_cand:
                    version_id = matching_cand["candidate_id"]

        if not version_id:
            continue

        instance_scores = val_result.get("instance_scores", [])
        seeds = val_result.get("seeds", [])
        accuracy = val_result.get("accuracy") or val_result.get("score")

        per_seed_scores = []
        if instance_scores and seeds:
            per_seed_scores = [
                {"seed": seeds[i] if i < len(seeds) else i, "score": float(score) if score is not None else 0.0}
                for i, score in enumerate(instance_scores)
            ]
        elif instance_scores:
            seeds_to_use = val_seeds if len(instance_scores) == len(val_seeds) else train_seeds
            per_seed_scores = [
                {"seed": seeds_to_use[i] if i < len(seeds_to_use) else i, "score": float(score) if score is not None else 0.0}
                for i, score in enumerate(instance_scores)
            ]

        per_seed_data["candidates"][version_id] = {
            "accuracy": float(accuracy) if accuracy is not None else None,
            "per_seed_scores": per_seed_scores,
            "num_seeds": len(per_seed_scores),
            "is_validation": True,
        }

    # Also extract per-seed scores from attempted_candidates/optimized_candidates (training phase)
    for cand_list_name in ["attempted_candidates", "optimized_candidates"]:
        for cand in job_results.get(cand_list_name, []):
            version_id = cand.get("version_id") or cand.get("id")
            if not version_id or version_id in per_seed_data["candidates"]:
                continue

            # Look for instance_scores or per_seed data
            instance_scores = cand.get("instance_scores", [])
            if not instance_scores:
                # Try from scoring field
                scoring = cand.get("scoring", {})
                instance_scores = scoring.get("instance_scores", [])

            accuracy = cand.get("accuracy") or cand.get("score") or cand.get("train_accuracy")

            per_seed_scores = []
            if instance_scores:
                per_seed_scores = [
                    {"seed": train_seeds[i] if i < len(train_seeds) else i, "score": float(score) if score is not None else 0.0}
                    for i, score in enumerate(instance_scores)
                ]

            per_seed_data["candidates"][version_id] = {
                "accuracy": _safe_acc(accuracy),
                "per_seed_scores": per_seed_scores,
                "num_seeds": len(per_seed_scores),
                "is_validation": False,
            }

    with open(per_seed_file, "w") as f:
        json.dump(per_seed_data, f, indent=2)
    print(f"✅ Saved per-seed scores to {per_seed_file} (candidates: {len(per_seed_data['candidates'])})")

    # 3. Save confusion matrices
    confusion_file = output_dir / "confusion_matrices.json"
    confusion_data = {}

    # Get Banking77 label names (77 intents)
    # These should be available from the task app or dataset
    label_names = [
        "activate_my_card",
        "age_limit",
        "apple_pay_or_google_pay",
        "atm_support",
        "automatic_top_up",
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "beneficiary_not_allowed",
        "cancel_transfer",
        "card_about_to_expire",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_not_working",
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "card_swallowed",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "change_pin",
        "compromised_card",
        "contactless_not_working",
        "country_support",
        "credit_card_payment_fee_charged",
        "credit_card_reporting_fraudulent",
        "credit_limit_change",
        "declined_card_payment",
        "declined_cash_withdrawal",
        "declined_transfer",
        "direct_debit_payment_not_recognised",
        "disposable_card_limits",
        "edit_personal_details",
        "exchange_charge",
        "exchange_rate",
        "exchange_via_app",
        "extra_charge_on_statement",
        "failed_transfer",
        "fiat_currency_support",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "lost_or_stolen_card",
        "order_physical_card",
        "passcode_forgotten",
        "pending_card_payment",
        "pending_cash_withdrawal",
        "pending_top_up",
        "pending_transfer",
        "pin_blocked",
        "receiving_money",
        "Refund_not_showing_up",
        "request_refund",
        "reverted_card_payment?",
        "reverted_card_payment_inquiry",
        "supported_cards_and_currencies",
        "terminate_account",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_list",
        "top_up_reverted",
        "transaction_charged_twice",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "verify_top_up",
        "virtual_card_not_working",
        "visa_or_mastercard",
        "why_verify_identity",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ]

    # Build confusion matrices from validation results
    # Note: We need to fetch rollout trajectories to get actual predicted/expected intents
    # For now, we'll build from what's available in validation results
    for val_result in validation_results:
        version_id = val_result.get("version_id")
        if not version_id:
            continue

        # Try to extract per-seed predictions if available
        instance_scores = val_result.get("instance_scores", [])
        seeds = val_result.get("seeds", [])

        # Build a simplified confusion matrix from instance scores
        # (We'd need actual trajectories for full confusion matrix)
        confusion_data[version_id] = {
            "accuracy": val_result.get("accuracy"),
            "note": "Full confusion matrix requires rollout trajectory data with predicted/expected intents per seed",
            "per_seed_correct": [
                {"seed": seeds[i] if i < len(seeds) else i, "correct": bool(score)}
                for i, score in enumerate(instance_scores)
            ],
        }

    with open(confusion_file, "w") as f:
        json.dump(confusion_data, f, indent=2)
    print(f"✅ Saved confusion matrices to {confusion_file}")

    # 4. Save pareto frontier
    pareto_file = output_dir / "pareto_frontier.json"

    def _safe_accuracy(c):
        """Extract accuracy as float, handling dict/None cases."""
        acc = c.get("accuracy", 0)
        if isinstance(acc, dict):
            # Try to extract score from dict
            return float(acc.get("score", 0) or acc.get("accuracy", 0) or 0)
        if acc is None:
            return 0.0
        return float(acc)

    pareto_data = {
        "frontier": [
            {
                "candidate_id": c["candidate_id"],
                "score": _safe_accuracy(c),
                "val_score": c.get("val_accuracy"),
            }
            for c in sorted(tracker.candidates, key=_safe_accuracy, reverse=True)[:10]
        ],
    }
    with open(pareto_file, "w") as f:
        json.dump(pareto_data, f, indent=2)
    print(f"✅ Saved pareto frontier to {pareto_file}")

    # 5. Save optimistic scores (best possible score if all improvements were realized)
    optimistic_file = output_dir / "optimistic_scores.json"
    optimistic_data = {
        "baseline_score": tracker.baseline_score,
        "best_achieved_score": tracker.best_score,
        "optimistic_score": tracker.best_score,  # Could be computed from per-seed improvements
        "improvement_potential": (
            (tracker.best_score - tracker.baseline_score) if tracker.baseline_score else None
        ),
    }
    with open(optimistic_file, "w") as f:
        json.dump(optimistic_data, f, indent=2)
    print(f"✅ Saved optimistic scores to {optimistic_file}")

    # 6. Save cost and time data
    cost_time_file = output_dir / "cost_and_time.json"
    elapsed_time = time.time() - tracker.start_time
    cost_data = {
        "total_time_seconds": elapsed_time,
        "total_time_minutes": elapsed_time / 60,
        "usage_data": tracker.usage_data,
        "finish_reason": tracker.finish_reason,
        "total_candidates": len(tracker.candidates),
    }
    with open(cost_time_file, "w") as f:
        json.dump(cost_data, f, indent=2)
    print(f"✅ Saved cost and time data to {cost_time_file}")

    # 7. Save baseline prompt details
    baseline_file = output_dir / "baseline_prompt.json"
    baseline_data = {
        "baseline_score": tracker.baseline_score,
        "baseline_instance_scores": tracker.baseline_instance_scores,
        "baseline_prompt": tracker.baseline_prompt,
        "num_seeds_evaluated": len(tracker.baseline_instance_scores),
        "seeds_passed": sum(1 for s in tracker.baseline_instance_scores if s > 0.5) if tracker.baseline_instance_scores else None,
        "seeds_failed": sum(1 for s in tracker.baseline_instance_scores if s <= 0.5) if tracker.baseline_instance_scores else None,
    }
    with open(baseline_file, "w") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"✅ Saved baseline prompt to {baseline_file}")

    # 8. Save pareto history (frontier updates over time)
    pareto_history_file = output_dir / "pareto_history.json"
    pareto_history_data = {
        "num_updates": len(tracker.pareto_history),
        "history": tracker.pareto_history,
    }
    with open(pareto_history_file, "w") as f:
        json.dump(pareto_history_data, f, indent=2)
    print(f"✅ Saved pareto history to {pareto_history_file} ({len(tracker.pareto_history)} updates)")

    # 9. Save seed analysis (disagreement between baseline and best)
    seed_analysis_file = output_dir / "seed_analysis.json"

    # Find best candidate (highest val_accuracy or train_accuracy)
    best_candidate = None
    best_score = -1
    for c in tracker.candidates:
        score = c.get("val_accuracy") or c.get("accuracy") or 0
        if isinstance(score, (int, float)) and score > best_score:
            best_score = score
            best_candidate = c

    # Calculate disagreement seeds
    disagreement_seeds = []
    baseline_wins = []
    best_wins = []

    if tracker.baseline_instance_scores and best_candidate and best_candidate.get("instance_scores"):
        best_instance_scores = best_candidate["instance_scores"]
        for i, (baseline_score, best_score_i) in enumerate(
            zip(tracker.baseline_instance_scores, best_instance_scores)
        ):
            seed_idx = train_seeds[i] if i < len(train_seeds) else i
            baseline_pass = baseline_score > 0.5 if baseline_score is not None else False
            best_pass = best_score_i > 0.5 if best_score_i is not None else False
            if baseline_pass != best_pass:
                disagreement_seeds.append({
                    "seed": seed_idx,
                    "baseline_score": baseline_score,
                    "best_score": best_score_i,
                    "baseline_pass": baseline_pass,
                    "best_pass": best_pass,
                })
                if baseline_pass and not best_pass:
                    baseline_wins.append(seed_idx)
                elif best_pass and not baseline_pass:
                    best_wins.append(seed_idx)

    seed_analysis_data = {
        "best_candidate_id": best_candidate["candidate_id"] if best_candidate else None,
        "best_candidate_score": best_score if best_score > 0 else None,
        "baseline_score": tracker.baseline_score,
        "num_disagreement_seeds": len(disagreement_seeds),
        "baseline_wins_seeds": baseline_wins,
        "best_wins_seeds": best_wins,
        "num_baseline_wins": len(baseline_wins),
        "num_best_wins": len(best_wins),
        "disagreement_seeds": disagreement_seeds,
    }
    with open(seed_analysis_file, "w") as f:
        json.dump(seed_analysis_data, f, indent=2)
    print(f"✅ Saved seed analysis to {seed_analysis_file} ({len(disagreement_seeds)} disagreements)")

    # 10. Save raw seed details (query, expected, predicted for each seed)
    seed_details_file = output_dir / "seed_details.json"

    def _extract_seed_details_from_rollouts(rollout_responses: list) -> list:
        """Extract seed details from rollout responses."""
        details = []
        for rollout in rollout_responses:
            if not isinstance(rollout, dict):
                continue
            trajectories = rollout.get("trajectories", [])
            metrics = rollout.get("metrics", {})
            for traj in trajectories:
                if not isinstance(traj, dict):
                    continue
                final = traj.get("final", {})
                obs = final.get("observation", {})
                steps = traj.get("steps", [])

                seed_idx = obs.get("index")
                query = obs.get("query", "")

                # Extract from step info
                expected_intent = None
                predicted_intent = None
                correct = None

                if steps:
                    last_step = steps[-1] if steps else {}
                    info = last_step.get("info", {})
                    expected_intent = info.get("expected_intent")
                    predicted_intent = info.get("predicted_intent")
                    correct = info.get("correct")

                # Fallback to metrics
                if correct is None:
                    correct = metrics.get("details", {}).get("correct")

                details.append({
                    "seed": seed_idx,
                    "query": query,
                    "expected_intent": expected_intent,
                    "predicted_intent": predicted_intent,
                    "correct": correct,
                    "reward": final.get("reward"),
                })
        return details

    # Extract seed details from baseline prompt rollouts
    baseline_seed_details = []
    if tracker.baseline_prompt and isinstance(tracker.baseline_prompt, dict):
        bp_meta = tracker.baseline_prompt.get("prompt_metadata", {})
        transformation = bp_meta.get("transformation", {})
        trans_meta = transformation.get("metadata", {})
        rollout_responses = trans_meta.get("rollout_responses", [])
        baseline_seed_details = _extract_seed_details_from_rollouts(rollout_responses)

    # Extract seed details from each candidate's rollouts
    candidate_seed_details = {}
    for c in tracker.candidates:
        raw_data = c.get("raw_data", {})
        if not raw_data:
            continue
        cand_id = c.get("candidate_id", "unknown")

        # Try to get rollout_responses from raw_data
        obj = raw_data.get("object", {})
        if isinstance(obj, dict):
            trans_meta = obj.get("metadata", {})
            rollout_responses = trans_meta.get("rollout_responses", [])
            if rollout_responses:
                candidate_seed_details[cand_id] = _extract_seed_details_from_rollouts(rollout_responses)

    seed_details_data = {
        "baseline": {
            "num_seeds": len(baseline_seed_details),
            "seeds": baseline_seed_details,
        },
        "candidates": candidate_seed_details,
    }
    with open(seed_details_file, "w") as f:
        json.dump(seed_details_data, f, indent=2)
    print(f"✅ Saved seed details to {seed_details_file} (baseline: {len(baseline_seed_details)} seeds, candidates: {len(candidate_seed_details)})")

    # 11. Save comprehensive analysis report
    report_file = output_dir / "analysis_report.md"
    with open(report_file, "w") as f:
        f.write("# Banking77 GEPA Demo - Comprehensive Analysis Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Baseline Score: {tracker.baseline_score:.4f}\n" if tracker.baseline_score else "- Baseline Score: N/A\n")
        f.write(f"- Best Score: {tracker.best_score:.4f}\n")
        f.write(f"- Total Candidates: {len(tracker.candidates)}\n")
        f.write(f"- Total Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)\n")
        f.write(f"- Finish Reason: {tracker.finish_reason or 'N/A'}\n\n")

        f.write("## Top 10 Candidates\n\n")
        top_candidates = sorted(tracker.candidates, key=_safe_accuracy, reverse=True)[:10]
        for i, cand in enumerate(top_candidates, 1):
            acc = _safe_accuracy(cand)
            val_acc = cand.get('val_accuracy')
            val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else str(val_acc or 'N/A')
            f.write(f"### Candidate #{i}: {cand['candidate_id']}\n\n")
            f.write(f"- Train Accuracy: {acc:.4f}\n")
            f.write(f"- Val Accuracy: {val_acc_str}\n")
            f.write(f"- Generation: {cand.get('generation', 'N/A')}\n")
            f.write(f"- Is Pareto: {cand.get('is_pareto', False)}\n\n")
            if cand.get("prompt_text"):
                f.write("**Prompt:**\n```\n")
                f.write(cand["prompt_text"][:500])
                f.write("\n```\n\n")

        f.write("\n## Files Generated\n\n")
        f.write("- `candidates.json`: All candidates with scores and prompts\n")
        f.write("- `per_seed_scores.json`: Per-seed scores for each candidate\n")
        f.write("- `confusion_matrices.json`: Confusion matrices (requires rollout data)\n")
        f.write("- `pareto_frontier.json`: Pareto-optimal candidates\n")
        f.write("- `optimistic_scores.json`: Optimistic score analysis\n")
        f.write("- `cost_and_time.json`: Cost and time breakdown\n")

    print(f"✅ Saved analysis report to {report_file}")


async def run():
    """Run the comprehensive Banking77 GEPA demo."""
    start_time = time.time()

    api_key = os.environ["SYNTH_API_KEY"]
    task_app_api_key = os.environ["ENVIRONMENT_API_KEY"]

    if not api_key:
        raise ValueError("SYNTH_API_KEY must be set")
    if not task_app_api_key:
        raise ValueError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")


    # Set environment variables for task app
    os.environ["SYNTH_API_KEY"] = api_key
    os.environ["ENVIRONMENT_API_KEY"] = task_app_api_key
    if os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    if os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Path to banking77 task app (in same directory as this script)
    task_app_path = Path(__file__).parent / "banking77_task_app.py"
    if not task_app_path.exists():
        raise FileNotFoundError(f"Task app not found at {task_app_path}")

    # Load TOML config - check for --test flag to use low-budget test config
    config_path = Path(__file__).parent / "banking77_gepa.toml"

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting InProcessTaskApp from {task_app_path}...")
    async with InProcessTaskApp(
        task_app_path=str(task_app_path),
        port=8120,
        auto_find_port=True,
        api_key=task_app_api_key,
        health_check_timeout=120.0,
    ) as task_app:
        print(f"InProcessTaskApp started at {task_app.url}")

        overrides = {
            "task_url": task_app.url,
            "prompt_learning.task_app_url": task_app.url,
        }

        job = PromptLearningJob.from_config(
            config_path=str(config_path),
            backend_url=BACKEND_URL_BASE,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            overrides=overrides,
        )

        job_id = job.submit()
        print(f"✓ Job submitted: {job_id}")

        # Create tracker
        tracker = Banking77DemoTracker()

        # Stream events via SSE (primary mechanism)
        async def stream_events():
            import httpx
            from synth_ai.sdk.api.train.utils import ensure_api_base

            api_base = ensure_api_base(BACKEND_URL_BASE)
            events_stream_url = f"{api_base}/prompt-learning/online/jobs/{job_id}/events/stream"
            job_url = f"{api_base}/prompt-learning/online/jobs/{job_id}"

            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0)) as client:
                try:
                    print("🔌 Connecting to SSE stream...")
                    async with client.stream(
                        "GET",
                        f"{events_stream_url}?since_seq=0",
                        headers={"X-API-Key": api_key, "Accept": "text/event-stream"},
                        timeout=httpx.Timeout(None, connect=30.0),
                    ) as response:
                        if response.status_code != 200:
                            raise Exception(f"SSE stream failed: {response.status_code}")
                        
                        print("✅ SSE stream connected, streaming events...")
                        buffer = ""
                        events_received = 0
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            buffer += chunk.decode("utf-8", errors="ignore")

                            while "\n\n" in buffer:
                                message, buffer = buffer.split("\n\n", 1)
                                if not message.strip():
                                    continue

                                for line in message.strip().split("\n"):
                                    if line.startswith("data: "):
                                        try:
                                            event_data = json.loads(line[6:])
                                            event_type = event_data.get("type", "").replace("[MASKED]", "gepa")
                                            event_payload = event_data.get("data", {})
                                            event = {
                                                "type": event_type,
                                                "data": event_payload if isinstance(event_payload, dict) else {},
                                            }
                                            tracker.update_from_event(event)
                                            events_received += 1
                                            
                                            # Log important events
                                            if events_received <= 5 or "candidate" in event_type.lower() or "complete" in event_type.lower():
                                                print(f"📨 Event #{events_received}: {event_type}")

                                            # Check for actual completion events (not proposal completion)
                                            if event_type in ("prompt.learning.completed", "prompt.learning.gepa.complete"):
                                                print(f"✅ Job completed via SSE (received {events_received} events)")
                                                tracker.status = "complete"
                                                # Wait a moment for backend to finalize
                                                await asyncio.sleep(2.0)
                                                return
                                            
                                            # Check for failure events
                                            if event_type in ("prompt.learning.failed", "prompt.learning.error"):
                                                print(f"❌ Job failed via SSE: {event_payload}")
                                                tracker.status = "failed"
                                                await asyncio.sleep(2.0)
                                                return
                                            
                                            # Don't exit on proposal.completed - that's just a phase completion, not job completion
                                        except json.JSONDecodeError:
                                            continue
                                    
                                    # Check if SSE stream closed unexpectedly (no more data)
                                    # This might mean the job is still running but SSE closed
                                    if not chunk and buffer == "":
                                        # Stream might have closed - check job status
                                        print(f"⚠️  SSE stream closed (received {events_received} events), checking job status...")
                                        break
                except Exception as e:
                    print(f"⚠️  SSE stream failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Don't raise - fall through to polling fallback

        # Use SSE as primary mechanism
        result = None
        try:
            await stream_events()
        except Exception as e:
            print(f"⚠️  SSE stream ended: {e}")
            import traceback
            traceback.print_exc()
        
        # Always check job status after SSE
        result = None
        try:
            result = await asyncio.to_thread(job.get_status)
            print(f"📊 Job status after SSE: {result.get('status')}")
        except Exception as e:
            # Handle SSL certificate errors on macOS Homebrew Python
            if "SSL" in str(e) or "certificate" in str(e).lower():
                print(f"⚠️  SSL error fetching status (macOS cert issue): {type(e).__name__}")
                if tracker.status == "complete":
                    print("   Job completed via SSE, continuing...")
                    result = {"status": "completed", "job_id": job_id}
                elif tracker.status == "failed":
                    print("   Job failed via SSE")
                    result = {"status": "failed", "job_id": job_id}
                else:
                    raise
            else:
                raise
        
        # If job is still running, wait for it to complete
        if result.get("status") not in ("succeeded", "completed", "failed", "cancelled"):
            print("⚠️  Job still running, polling until completion...")
            print("   (This may take a while - validation phase evaluates candidates on 200 seeds)")
            # Suppress poll logs
            import contextlib
            with contextlib.suppress(Exception):
                import click
                original_echo = click.echo
                def filtered_echo(msg="", **kwargs):
                    if isinstance(msg, str) and "[poll]" in msg:
                        return
                    return original_echo(msg, **kwargs)
                click.echo = filtered_echo
            
            try:
                result = await asyncio.to_thread(job.poll_until_complete, timeout=7200.0, interval=5.0)
            finally:
                with contextlib.suppress(Exception):
                    import click
                    click.echo = original_echo
        
        # Check if job failed
        if result.get("status") == "failed":
            error_msg = result.get("error") or result.get("error_message") or "Unknown error"
            print(f"\n❌ Job failed: {error_msg}")
            print(f"📋 Full job status: {json.dumps(result, indent=2)}")
        
        print(f"✅ Job final status: {result.get('status')}")

        # Fetch detailed results
        from synth_ai.sdk.api.train.utils import ensure_api_base
        from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
        import httpx

        api_base = ensure_api_base(BACKEND_URL_BASE)

        # Try to fetch results, but handle SSL errors gracefully
        prompt_results = None
        scoring_summary = None
        job_results = None

        try:
            client = PromptLearningClient(api_base, api_key)
            prompt_results = await client.get_prompts(job_id)
            scoring_summary = await client.get_scoring_summary(job_id)
        except Exception as e:
            if "SSL" in str(e) or "certificate" in str(e).lower():
                print(f"⚠️  SSL error fetching detailed results (macOS cert issue)")
                print(f"   Job completed successfully but cannot fetch results due to SSL.")
                print(f"   To fix: Run 'pip install certifi' and set SSL_CERT_FILE env var")
                print(f"   Job ID: {job_id}")
                print(f"\n🎉 Demo completed! Job {job_id} finished successfully.")
                return
            else:
                raise

        # Get job results
        job_results = await asyncio.to_thread(job.get_results)

        # Merge metadata from job status into job_results for access to seeds etc.
        job_status_metadata = result.get("metadata", {})
        if not job_results.get("config"):
            job_results["config"] = job_status_metadata.get("prompt_initial_snapshot", {}).get("raw_config", {})
        if not job_results.get("train_seeds"):
            pl_config = job_results.get("config", {}).get("prompt_learning", {})
            job_results["train_seeds"] = pl_config.get("train_seeds", [])
            gepa_eval = pl_config.get("gepa", {}).get("evaluation", {})
            if not job_results["train_seeds"]:
                job_results["train_seeds"] = gepa_eval.get("train_seeds", [])
            if not job_results.get("val_seeds"):
                job_results["val_seeds"] = gepa_eval.get("val_seeds", [])

        # Extract baseline_score from job_results (authoritative source)
        tracker.baseline_score = (
            tracker.baseline_score  # Keep if captured from events
            or job_results.get("baseline_score")
            or scoring_summary.get("baseline_score")
            or scoring_summary.get("baseline_accuracy")
        )

        # Extract baseline prompt and instance scores
        tracker.baseline_prompt = job_results.get("best_prompt")  # Initial prompt before optimization
        tracker.baseline_instance_scores = (
            job_results.get("baseline_instance_scores", [])
            or scoring_summary.get("baseline_instance_scores", [])
        )

        # Also try to get validation results which may have baseline data
        validation_results = job_results.get("validation_results", [])
        for vr in validation_results:
            if vr.get("is_baseline") or vr.get("rank") == -1:
                if not tracker.baseline_instance_scores:
                    tracker.baseline_instance_scores = vr.get("instance_scores", [])
                if tracker.baseline_score is None:
                    tracker.baseline_score = vr.get("accuracy")

        # Extract candidates from job_results (authoritative source, like redactle example)
        print("📊 Extracting candidates from job_results...")
        attempted_candidates = job_results.get("attempted_candidates", [])
        optimized_candidates = job_results.get("optimized_candidates", [])
        # Combine and deduplicate by version_id (like redactle)
        # Note: attempted_candidates may not have version_id, use index as fallback
        all_candidates_dict = {}
        for idx, cand in enumerate(attempted_candidates):
            version_id = cand.get("version_id") or cand.get("id") or f"attempted_{idx}"
            all_candidates_dict[version_id] = cand

        # Overwrite with optimized candidates (they have more complete data)
        for cand in optimized_candidates:
            version_id = cand.get("version_id") or cand.get("id") or f"optimized_{len(all_candidates_dict)}"
            all_candidates_dict[version_id] = cand
        
        # Build lookup from prompt_results for fallback prompt extraction
        prompt_lookup = {}
        if isinstance(prompt_results, list):
            for pr in prompt_results:
                if isinstance(pr, dict):
                    vid = pr.get("version_id") or pr.get("id")
                    if vid:
                        prompt_lookup[vid] = pr
        elif isinstance(prompt_results, dict):
            # Maybe it's a dict with prompts
            for vid, pr in prompt_results.items():
                if isinstance(pr, dict):
                    prompt_lookup[vid] = pr

        # Convert to tracker format
        tracker.candidates = []
        for cand in all_candidates_dict.values():
            version_id = cand.get("version_id") or cand.get("id") or f"c{len(tracker.candidates)}"

            # Try to extract prompt from candidate data
            # Priority: 1. messages, 2. object.prompt_sections, 3. prompt_text field, 4. best_prompt
            prompt_text = None

            # 1. Try direct messages field (added from archive_summary)
            messages = cand.get("messages")
            if messages and isinstance(messages, list):
                parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content") or msg.get("pattern", "")
                        if content:
                            parts.append(f"[{role.upper()}]: {content}")
                if parts:
                    prompt_text = "\n\n".join(parts)

            # 2. Try object.prompt_sections (for template payloads)
            if not prompt_text:
                obj = cand.get("object", {})
                if isinstance(obj, dict):
                    prompt_sections = obj.get("prompt_sections", [])
                    if prompt_sections and isinstance(prompt_sections, list):
                        parts = []
                        for sec in prompt_sections:
                            if isinstance(sec, dict):
                                role = sec.get("role", "unknown")
                                content = sec.get("content") or sec.get("pattern", "")
                                if content:
                                    parts.append(f"[{role.upper()}]: {content}")
                        if parts:
                            prompt_text = "\n\n".join(parts)

            # 3. For transformations, show the transformation details
            if not prompt_text:
                obj = cand.get("object", {})
                payload_kind = cand.get("payload_kind") or cand.get("type")
                # Check if object contains transformation fields
                if isinstance(obj, dict) and (obj.get("text_replacements") or obj.get("example_injections")):
                    text_replacements = obj.get("text_replacements", [])
                    example_injections = obj.get("example_injections", [])
                    if text_replacements or example_injections:
                        parts = ["[TRANSFORMATION]"]
                        for tr in text_replacements:
                            if isinstance(tr, dict):
                                old = tr.get("old_text", "")[:50]
                                new = tr.get("new_text", "")[:100]
                                role = tr.get("apply_to_role", "all")
                                parts.append(f"Replace in {role}: '{old}...' -> '{new}...'")
                        for inj in example_injections:
                            if isinstance(inj, dict):
                                after_role = inj.get("insert_after_role", "")
                                examples = inj.get("examples", [])
                                parts.append(f"Inject {len(examples)} examples after {after_role}")
                        prompt_text = "\n".join(parts)

            # 4. Try prompt_text field directly
            if not prompt_text:
                prompt_text = cand.get("prompt_text")

            # 5. Try from prompt_results lookup
            if not prompt_text and version_id in prompt_lookup:
                pr = prompt_lookup[version_id]
                prompt_text = pr.get("prompt_text") or (pr.get("object", {}) or {}).get("prompt_text")
                if not prompt_text:
                    pr_messages = pr.get("messages", [])
                    if pr_messages and isinstance(pr_messages, list):
                        parts = []
                        for msg in pr_messages:
                            if isinstance(msg, dict):
                                role = msg.get("role", "")
                                content = msg.get("content", "")
                                if content:
                                    parts.append(f"[{role.upper()}]: {content}")
                        if parts:
                            prompt_text = "\n\n".join(parts)

            # Extract seed_eval_info and instance_scores
            seed_eval_info = cand.get("seed_eval_info", {})
            instance_scores = cand.get("instance_scores", [])
            if not instance_scores and seed_eval_info:
                instance_scores = seed_eval_info.get("instance_scores", [])

            candidate = {
                "candidate_id": version_id,
                "accuracy": cand.get("accuracy") or cand.get("score") or cand.get("train_accuracy", 0),
                "val_accuracy": cand.get("val_accuracy") or cand.get("full_score"),
                "train_accuracy": cand.get("train_accuracy") or cand.get("accuracy"),
                "prompt_text": prompt_text,
                "generation": cand.get("generation"),
                "parent_id": cand.get("parent_id"),
                "is_pareto": cand.get("is_pareto", False),
                "instance_scores": instance_scores,
                "seed_eval_info": seed_eval_info if seed_eval_info else None,
                "timestamp": time.time(),
                "raw_data": cand,
            }
            tracker.candidates.append(candidate)
            acc = candidate["accuracy"]
            if acc and isinstance(acc, (int, float)) and acc > tracker.best_score:
                tracker.best_score = acc

        print(f"✓ Extracted {len(tracker.candidates)} candidates from job_results")

        # Try to fetch rollout trajectories for confusion matrices
        # Note: This may not be available via the SDK, so we'll use what's in validation_results
        trajectories_data = {}
        try:
            async with httpx.AsyncClient() as http_client:
                # Try to fetch trajectories endpoint if available
                traj_url = f"{api_base}/prompt-learning/online/jobs/{job_id}/trajectories"
                try:
                    resp = await http_client.get(traj_url, headers={"X-API-Key": api_key}, timeout=30.0)
                    if resp.status_code == 200:
                        trajectories_data = resp.json()
                except Exception:
                    # Trajectories endpoint may not exist - that's okay
                    pass
        except Exception:
            pass

        # Save comprehensive results
        try:
            save_comprehensive_results(output_dir, tracker, job_results, prompt_results, scoring_summary)
        except Exception as e:
            print(f"⚠️  Error saving some results: {e}")
            import traceback
            traceback.print_exc()
            # Still try to save basic results
            try:
                basic_results_file = output_dir / "basic_results.json"
                with open(basic_results_file, "w") as f:
                    json.dump({
                        "job_id": job_id,
                        "baseline_score": tracker.baseline_score,
                        "best_score": tracker.best_score,
                        "total_candidates": len(tracker.candidates),
                        "status": tracker.status,
                        "error": str(e),
                    }, f, indent=2)
                print(f"✅ Saved basic results to {basic_results_file}")
            except Exception:
                pass

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        
        # List all saved files
        print("\n📁 Saved files:")
        for file in sorted(output_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size
                print(f"   - {file.name} ({size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(run())

