import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "banking77.json"


class EnvSpec(BaseModel):
    seed: Optional[int] = Field(default=0)


class PolicyConfig(BaseModel):
    prompt_template: str
    inference_url: str
    model: Optional[str] = None


class PolicySpec(BaseModel):
    config: PolicyConfig


class RolloutRequest(BaseModel):
    env: EnvSpec
    policy: PolicySpec


class RolloutStep(BaseModel):
    reward: float


class RolloutTrajectory(BaseModel):
    steps: list[RolloutStep]


class RolloutResponse(BaseModel):
    metrics: dict
    trajectories: list[RolloutTrajectory]

    @staticmethod
    def single_reward(reward: float) -> "RolloutResponse":
        return RolloutResponse(
            metrics={"mean_return": reward},
            trajectories=[RolloutTrajectory(steps=[RolloutStep(reward=reward)])],
        )


def load_dataset() -> list[tuple[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples") or []
    return [(row["text"], row["label"]) for row in samples]


def load_labels() -> set[str]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels") or []
    return set(label.strip().lower() for label in labels)


DATASET = load_dataset()
LABEL_SET = load_labels()

ENVIRONMENT_API_KEY = os.getenv("ENVIRONMENT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Polyglot Task App (Python)", version="1.0.0")


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if ENVIRONMENT_API_KEY:
        if x_api_key != ENVIRONMENT_API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/task_info", dependencies=[Depends(require_api_key)])
def task_info() -> dict:
    return {
        "name": "banking77",
        "description": "Intent classification for banking queries",
        "observation_space": {"text": "string"},
        "action_space": {"label": "string"},
        "reward_range": [0, 1],
        "count": len(DATASET),
    }


def build_inference_request(inference_url: str, prompt_template: str, text: str) -> tuple[str, dict, dict]:
    parsed = urlparse(inference_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    query = parsed.query

    # Default to OpenAI-style chat completions
    api_path = f"{path}/chat/completions"
    final_url = f"{base}{api_path}"
    if query:
        final_url = f"{final_url}?{query}"

    model = None
    for part in query.split("&"):
        if part.startswith("model="):
            model = part.split("=", 1)[1]
            break

    # Inject a constrained label list to improve accuracy (user prompt gets labels too)
    label_list = ", ".join(sorted(LABEL_SET))
    system_prompt = (
        "You are an intent classifier for banking queries. "
        f"Valid labels are: {label_list}. Respond with exactly one label."
    )
    user_prompt = (
        prompt_template.replace("{{text}}", text)
        + f"\n\nValid labels: {label_list}\nRespond with exactly one label."
    )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    headers: dict = {"Content-Type": "application/json"}
    token = None
    if "openai" in parsed.netloc and OPENAI_API_KEY:
        token = OPENAI_API_KEY
    elif "groq" in parsed.netloc and GROQ_API_KEY:
        token = GROQ_API_KEY
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return final_url, headers, body


async def call_llm(inference_url: str, prompt_template: str, text: str) -> str:
    url, headers, body = build_inference_request(inference_url, prompt_template, text)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content.strip()


def compute_reward(predicted: str, label: str) -> float:
    if not predicted:
        return 0.0
    text = predicted.strip().lower()
    target = label.strip().lower()
    # Exact label containment
    if target in text:
        return 1.0
    # If the model returned a known label anywhere, count it
    for lbl in LABEL_SET:
        if lbl in text:
            return 1.0 if lbl == target else 0.0
    # Fallback to first token match
    first_token = text.split()[0]
    return 1.0 if first_token == target else 0.0


@app.post("/rollout", response_model=RolloutResponse, dependencies=[Depends(require_api_key)])
async def rollout(req: RolloutRequest) -> RolloutResponse:
    seed = req.env.seed or 0
    if seed < 0 or seed >= len(DATASET):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid seed")

    text, label = DATASET[seed]
    predicted = await call_llm(
        req.policy.config.inference_url,
        req.policy.config.prompt_template,
        text,
    )
    reward = compute_reward(predicted, label)
    return RolloutResponse.single_reward(reward)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=False,
    )
