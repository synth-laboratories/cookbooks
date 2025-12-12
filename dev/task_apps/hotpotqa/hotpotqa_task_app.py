"""HotpotQA multi-hop question answering task app for Synth prompt optimization benchmarks."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

from synth_ai.sdk.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.sdk.task.auth import is_api_key_header_authorized, normalize_environment_api_key
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.sdk.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.sdk.task.rubrics import Rubric, load_rubric
from synth_ai.sdk.task.server import ProxyConfig, RubricBundle, TaskAppConfig, create_task_app, run_task_app
from synth_ai.sdk.task.vendors import normalize_vendor_keys


def _compute_repo_root() -> Path:
    p = Path(__file__).resolve()
    parents = list(p.parents)
    if len(parents) >= 4:
        return parents[3]
    if "/opt/synth_ai_repo" in os.getenv("PYTHONPATH", "") or Path("/opt/synth_ai_repo/synth_ai").exists():
        return Path("/opt/synth_ai_repo")
    return Path.cwd()


REPO_ROOT = _compute_repo_root()

# Dataset configuration
DATASET_NAME = os.getenv("HOTPOTQA_DATASET_NAME", "hotpotqa/hotpot_qa")
DEFAULT_SPLIT = "train"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "validation")

print(
    f"[hotpotqa_task_app] Module loaded: DATASET_NAME={DATASET_NAME}",
    flush=True,
)


class HotpotQADataset:
    """Lazy Hugging Face dataset loader for HotpotQA."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split: {split}. Available: {AVAILABLE_SPLITS}")
        if split not in self._cache:
            try:
                from datasets import load_dataset as _load_dataset

                print(
                    f"[HotpotQADataset] Loading dataset '{DATASET_NAME}' split '{split}'",
                    flush=True,
                )

                # HotpotQA has 'distractor' and 'fullwiki' subsets - use distractor (easier)
                ds = _load_dataset(
                    DATASET_NAME,
                    "distractor",
                    split=split,
                    trust_remote_code=True,
                )

                self._cache[split] = ds
                print(
                    f"[HotpotQADataset] Successfully loaded {len(ds)} examples from '{DATASET_NAME}' split '{split}'",
                    flush=True,
                )
            except Exception as exc:
                import traceback
                error_details = traceback.format_exc()
                print(
                    f"[HotpotQADataset] Dataset load failed: {exc}\n{error_details}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Dataset preparation failed: {split}: Failed to load HotpotQA dataset. "
                    f"Dataset: {DATASET_NAME} | Split: {split} | Error: {exc}"
                ) from exc
        return self._cache[split]

    def ensure_ready(self, splits: Sequence[str]) -> None:
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        dataset = self._load_split(split)
        return len(dataset)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"HotpotQA split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        # Build context from supporting facts
        context_parts = []
        for title, sentences in zip(row.get("context", {}).get("title", []),
                                     row.get("context", {}).get("sentences", [])):
            context_parts.append(f"**{title}**\n" + " ".join(sentences))
        context = "\n\n".join(context_parts)

        return {
            "index": idx,
            "split": split,
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "context": context,
            "type": str(row.get("type", "")),
            "level": str(row.get("level", "")),
        }


hotpotqa_router = APIRouter()


HOTPOTQA_DATASET_SPEC = TaskDatasetSpec(
    id="hotpotqa",
    name="HotpotQA Multi-Hop QA",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Multi-hop question answering requiring reasoning across multiple passages.",
)


async def call_chat_completion(
    policy_config: dict[str, Any],
    placeholders: dict[str, Any],
    default_messages: list[dict[str, str]],
    api_key: str | None = None,
    http_client: Any | None = None,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    """Call the chat completion API and return response."""
    import socket
    from urllib.parse import urlparse, urlunparse

    model_val = policy_config.get("model")
    if not isinstance(model_val, str) or not model_val.strip():
        raise HTTPException(status_code=400, detail="Missing policy field: model")

    inference_url_raw = policy_config.get("inference_url")
    api_base_raw = policy_config.get("api_base")
    base_url_raw = policy_config.get("base_url")

    if inference_url_raw:
        route_base = str(inference_url_raw).strip()
    else:
        route_base = (api_base_raw or "").strip() or (base_url_raw or "").strip()

    if not route_base:
        raise HTTPException(status_code=400, detail="Missing policy field: inference_url")

    model = policy_config["model"].strip()
    provider = str(policy_config.get("provider", "")).strip() or "groq"

    def _normalize_chat_url(url: str) -> str:
        u = (url or "").rstrip("/")
        if not u:
            return "/chat/completions"
        parsed = urlparse(u)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
            return u
        if "/v1/" in path and not path.endswith("/v1"):
            new_path = f"{path}/chat/completions"
            return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))
        if path.endswith("/v1"):
            new_path = f"{path}/chat/completions"
        else:
            new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
        return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))

    inference_url = _normalize_chat_url(str(route_base))
    temperature = policy_config.get("temperature", 0.0)
    max_tokens = policy_config.get("max_completion_tokens", 512)

    messages = []
    for msg_template in default_messages:
        role = msg_template.get("role", "user")
        pattern = msg_template.get("pattern", "")
        content = pattern.format(**placeholders)
        messages.append({"role": role, "content": content})

    headers: dict[str, str] = {"Content-Type": "application/json"}
    lowered = route_base.lower()
    is_provider_host = ("api.openai.com" in lowered) or ("api.groq.com" in lowered)

    if api_key:
        if is_provider_host:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["X-API-Key"] = api_key

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if temperature != 0.0:
        payload["temperature"] = temperature

    print(f"[TASK_APP] POLICY ROUTE -> {inference_url}", flush=True)

    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    response_json: dict[str, Any] | None = None
    try:
        import aiohttp
        is_aiohttp = isinstance(http_client, aiohttp.ClientSession)

        if is_aiohttp:
            async with http_client.post(inference_url, json=payload, headers=headers) as response:
                status_code = response.status
                if status_code != 200:
                    error_text = await response.text()
                    raise HTTPException(status_code=status_code, detail=f"API error: {error_text[:200]}")
                response_json = await response.json()
        else:
            response = await http_client.post(inference_url, json=payload, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"API error: {response.text[:200]}")
            response_json = response.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Request failed: {e}")

    if response_json is None:
        raise HTTPException(status_code=502, detail="No response data")

    response_text = ""
    if "choices" in response_json and len(response_json["choices"]) > 0:
        choice = response_json["choices"][0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

    return response_text, response_json, []


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, strip, remove articles)."""
    import re
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: HotpotQADataset = fastapi_request.app.state.hotpotqa_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "question": sample["question"],
        "context": sample["context"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "question": sample["question"],
        "context": sample["context"],
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a research assistant that answers multi-hop questions. "
                "Read the passages carefully and provide a short, precise answer. "
                "Format: Answer: <your answer>"
            ),
        },
        {
            "role": "user",
            "pattern": "Question: {question}\n\nPassages:\n{context}\n\nProvide the answer.",
        },
    ]

    api_key = (
        fastapi_request.headers.get("X-API-Key")
        or fastapi_request.headers.get("x-api-key")
        or None
    )

    http_client = getattr(fastapi_request.app.state, "http_client", None)

    response_text, response_json, _ = await call_chat_completion(
        request.policy.config or {},
        placeholders,
        default_messages,
        api_key=api_key,
        http_client=http_client,
    )

    # Extract answer from response
    predicted_answer = response_text.strip()
    if "Answer:" in predicted_answer:
        predicted_answer = predicted_answer.split("Answer:")[-1].strip()

    expected_answer = sample["answer"]

    # Compute F1 score (standard for HotpotQA)
    f1_score = compute_f1(predicted_answer, expected_answer)
    exact_match = float(normalize_answer(predicted_answer) == normalize_answer(expected_answer))

    # Use F1 as reward
    reward = f1_score

    print(
        f"[HOTPOTQA_ROLLOUT] seed={seed} expected={expected_answer[:50]}... predicted={predicted_answer[:50]}... f1={f1_score:.3f}",
        flush=True,
    )

    step = RolloutStep(
        obs=observation,
        tool_calls=[],
        reward=reward,
        done=True,
        info={
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "f1_score": f1_score,
            "exact_match": exact_match,
        },
    )

    trajectory = RolloutTrajectory(
        env_id=f"hotpotqa::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step],
        final={"observation": observation, "reward": reward},
        length=1,
        inference_url=str((request.policy.config or {}).get("inference_url") or ""),
    )

    metrics = RolloutMetrics(
        episode_returns=[reward],
        mean_return=reward,
        num_steps=1,
        num_episodes=1,
        outcome_score=reward,
        events_score=reward,
        details={"f1_score": f1_score, "exact_match": exact_match},
    )

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=1,
        trace=None,
    )


def build_dataset() -> tuple[TaskDatasetRegistry, HotpotQADataset]:
    registry = TaskDatasetRegistry()
    dataset = HotpotQADataset()
    registry.register(HOTPOTQA_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "hotpotqa",
            "name": "HotpotQA Multi-Hop QA",
            "version": "1.0.0",
            "action_space": {
                "type": "text",
                "description": "Answer multi-hop questions by reasoning across passages.",
            },
        },
        environment="hotpotqa",
        dataset=HOTPOTQA_DATASET_SPEC.model_dump(),
        rubric={"version": "1", "criteria_count": 1, "source": "inline"},
        inference={"supports_proxy": True},
        limits={"max_turns": 1},
        task_metadata={"format": "text"},
    )


def describe_taskset(dataset: HotpotQADataset) -> Mapping[str, Any]:
    return {
        **HOTPOTQA_DATASET_SPEC.model_dump(),
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: HotpotQADataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset={**base_info.dataset, "split": sample["split"], "index": sample["index"]},
            rubric=base_info.rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata={
                **base_info.task_metadata,
                "question": sample["question"],
                "expected_answer": sample["answer"],
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric({
        "version": "1",
        "goal_text": "Answer multi-hop questions accurately.",
        "aggregation": "weighted_sum",
        "criteria": [{"id": "answer_accuracy", "description": "Provide correct answer.", "weight": 1.0}],
    }),
)


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    print("[hotpotqa_task_app] Preloading dataset splits...", flush=True)
    try:
        dataset.ensure_ready(AVAILABLE_SPLITS)
        print(f"[hotpotqa_task_app] Dataset preloaded: {[dataset.size(s) for s in AVAILABLE_SPLITS]}", flush=True)
    except Exception as exc:
        print(f"[hotpotqa_task_app] WARNING: Dataset preload failed: {exc}", flush=True)

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
    )

    async def startup_http_client(app: Any) -> None:
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=30.0)
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            app.state.http_client = aiohttp.ClientSession(timeout=timeout, connector=connector)
            print("[hotpotqa_task_app] Created aiohttp client session", flush=True)
        except Exception as exc:
            print(f"[hotpotqa_task_app] WARNING: Failed to create http client: {exc}", flush=True)
            app.state.http_client = None

    async def shutdown_http_client(app: Any) -> None:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            try:
                await http_client.close()
            except Exception:
                pass

    config = TaskAppConfig(
        app_id="hotpotqa",
        name="HotpotQA Multi-Hop Question Answering Task",
        description="HotpotQA dataset task app for multi-hop question answering.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=OUTCOME_RUBRIC),
        proxy=proxy_config,
        routers=(hotpotqa_router,),
        app_state={"hotpotqa_dataset": dataset},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="hotpotqa",
        description="HotpotQA multi-hop question answering task app.",
        config_factory=build_config,
        aliases=("hotpot",),
        modal=ModalDeploymentConfig(
            app_name="synth-hotpotqa",
            pip_packages=(
                "datasets>=2.14.0",
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "aiohttp>=3.9.0",
            ),
        ),
    )
)


def fastapi_app():
    """Return the FastAPI application."""
    with contextlib.suppress(Exception):
        load_dotenv(str(REPO_ROOT / ".env"), override=False)

    app = create_task_app(build_config())
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the HotpotQA task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
