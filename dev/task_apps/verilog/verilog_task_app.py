"""VerilogEval spec-to-RTL task app for Synth prompt optimization benchmarks.

This is an agentic task where the model uses tools (write_file, compile, simulate, submit)
to implement and verify Verilog hardware designs.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import shutil
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
DATASET_NAME = os.getenv("VERILOG_DATASET_NAME", "dakies/nvlabs-verilogeval-v2-spec-to-rtl")
DEFAULT_SPLIT = "test"
AVAILABLE_SPLITS: tuple[str, ...] = ("test",)  # VerilogEval v2 only has test split

# Tool names
TOOL_WRITE_FILE = "write_file"
TOOL_COMPILE = "compile"
TOOL_SIMULATE = "simulate"
TOOL_SUBMIT = "submit"

# Max agentic steps
MAX_STEPS = 10

print(
    f"[verilog_task_app] Module loaded: DATASET_NAME={DATASET_NAME}",
    flush=True,
)


class VerilogEvalDataset:
    """Lazy Hugging Face dataset loader for VerilogEval v2."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split: {split}. Available: {AVAILABLE_SPLITS}")
        if split not in self._cache:
            try:
                from datasets import load_dataset as _load_dataset

                print(
                    f"[VerilogEvalDataset] Loading dataset '{DATASET_NAME}' split '{split}'",
                    flush=True,
                )

                ds = _load_dataset(
                    DATASET_NAME,
                    split=split,
                    trust_remote_code=True,
                )

                self._cache[split] = ds
                print(
                    f"[VerilogEvalDataset] Successfully loaded {len(ds)} examples from '{DATASET_NAME}' split '{split}'",
                    flush=True,
                )
            except Exception as exc:
                import traceback
                error_details = traceback.format_exc()
                print(
                    f"[VerilogEvalDataset] Dataset load failed: {exc}\n{error_details}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Dataset preparation failed: {split}: Failed to load VerilogEval dataset. "
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
            raise RuntimeError(f"VerilogEval split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        return {
            "index": idx,
            "split": split,
            "problem_id": str(row.get("problem_id", f"problem_{idx}")),
            "prompt": str(row.get("prompt", "")),
            "test": str(row.get("test", "")),  # testbench
            "ref": str(row.get("ref", "")),    # reference solution
        }


verilog_router = APIRouter()


VERILOG_DATASET_SPEC = TaskDatasetSpec(
    id="verilog",
    name="VerilogEval v2 Spec-to-RTL",
    version="2.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="VerilogEval v2 specification-to-RTL translation tasks.",
)


class VerilogWorkspace:
    """Manages a temporary workspace for Verilog compilation and simulation."""

    def __init__(self, problem_id: str, prompt: str, testbench: str, ref_solution: str):
        self.problem_id = problem_id
        self.prompt = prompt
        self.testbench = testbench
        self.ref_solution = ref_solution
        self.workspace_dir = Path(tempfile.mkdtemp(prefix=f"verilog_{problem_id}_"))
        self.files: dict[str, str] = {}
        self.last_compile_output: str | None = None
        self.last_simulate_output: str | None = None
        self.submitted = False
        self.passed = False

        # Write initial files
        self._setup_workspace()

    def _setup_workspace(self):
        """Set up the workspace with initial files."""
        # Create incomplete module template
        module_content = f"""module TopModule();
    // TODO: Implement the module based on the specification below
    /*
    Specification:
    {self.prompt.strip()}
    */
endmodule"""

        # Write files
        (self.workspace_dir / "TopModule.v").write_text(module_content)
        (self.workspace_dir / f"{self.problem_id}_tb.v").write_text(self.testbench)
        (self.workspace_dir / "RefModule.v").write_text(self.ref_solution)

        self.files = {
            "TopModule.v": module_content,
            f"{self.problem_id}_tb.v": self.testbench,
            "RefModule.v": self.ref_solution,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write content to a file in the workspace."""
        try:
            file_path = self.workspace_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            self.files[path] = content
            return {"ok": True, "message": f"Wrote {len(content)} bytes to {path}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def compile(self, sources: list[str] | None = None, testbench: str | None = None) -> dict[str, Any]:
        """Compile Verilog sources with iverilog."""
        try:
            # Default sources
            if sources is None:
                sources = ["TopModule.v"]
            if testbench is None:
                testbench = f"{self.problem_id}_tb.v"

            # Build compile command
            all_sources = sources + [testbench]
            source_paths = [str(self.workspace_dir / s) for s in all_sources]
            output_path = str(self.workspace_dir / "a.out")

            cmd = ["iverilog", "-o", output_path] + source_paths

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_dir),
            )

            self.last_compile_output = result.stdout + result.stderr

            if result.returncode == 0:
                return {"ok": True, "output": self.last_compile_output, "binary": "a.out"}
            else:
                return {"ok": False, "output": self.last_compile_output, "error": "Compilation failed"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Compilation timed out"}
        except FileNotFoundError:
            return {"ok": False, "error": "iverilog not found - ensure Icarus Verilog is installed"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def simulate(self, binary: str | None = None) -> dict[str, Any]:
        """Run vvp on compiled binary."""
        try:
            if binary is None:
                binary = "a.out"

            binary_path = self.workspace_dir / binary
            if not binary_path.exists():
                return {"ok": False, "error": f"Binary '{binary}' not found. Run compile first."}

            result = subprocess.run(
                ["vvp", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.workspace_dir),
            )

            self.last_simulate_output = result.stdout + result.stderr

            # Check for pass/fail patterns
            stdout = self.last_simulate_output
            passed = (
                "ALL_TESTS_PASSED" in stdout
                or ("Mismatches: 0 " in stdout and "samples" in stdout)
                or ("no mismatches" in stdout.lower() and "errors" not in stdout.lower())
            )

            return {
                "ok": True,
                "output": self.last_simulate_output,
                "passed": passed,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Simulation timed out"}
        except FileNotFoundError:
            return {"ok": False, "error": "vvp not found - ensure Icarus Verilog is installed"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def submit(self) -> dict[str, Any]:
        """Submit solution for grading."""
        self.submitted = True

        # Check if simulation passed
        if self.last_simulate_output:
            stdout = self.last_simulate_output
            self.passed = (
                "ALL_TESTS_PASSED" in stdout
                or ("Mismatches: 0 " in stdout and "samples" in stdout)
                or ("no mismatches" in stdout.lower() and "errors" not in stdout.lower())
            )
        else:
            self.passed = False

        return {
            "ok": True,
            "submitted": True,
            "passed": self.passed,
            "message": "Tests passed!" if self.passed else "Tests failed",
        }

    def cleanup(self):
        """Clean up the workspace directory."""
        try:
            shutil.rmtree(self.workspace_dir)
        except Exception:
            pass


async def call_chat_completion_with_tools(
    policy_config: dict[str, Any],
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    api_key: str | None = None,
    http_client: Any | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    """Call the chat completion API with tools and return response."""
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
        "tools": tools,
        "tool_choice": "auto",
    }
    if temperature != 0.0:
        payload["temperature"] = temperature

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
    tool_calls = []
    if "choices" in response_json and len(response_json["choices"]) > 0:
        choice = response_json["choices"][0]
        message = choice.get("message", {})
        response_text = message.get("content", "") or ""

        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    }
                })

    return response_text, tool_calls, response_json


def build_verilog_tools() -> list[dict[str, Any]]:
    """Build the tool schemas for Verilog operations."""
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_WRITE_FILE,
                "description": "Write content to a Verilog file in the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path (e.g., TopModule.v)"},
                        "content": {"type": "string", "description": "File content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_COMPILE,
                "description": "Compile Verilog sources with iverilog",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of source files to compile (default: [TopModule.v])",
                        },
                        "testbench": {"type": "string", "description": "Testbench file (optional)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_SIMULATE,
                "description": "Run vvp simulation on compiled binary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "binary": {"type": "string", "description": "Binary file to simulate (default: a.out)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_SUBMIT,
                "description": "Submit solution for final grading",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: VerilogEvalDataset = fastapi_request.app.state.verilog_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)

    # Create workspace
    workspace = VerilogWorkspace(
        problem_id=sample["problem_id"],
        prompt=sample["prompt"],
        testbench=sample["test"],
        ref_solution=sample["ref"],
    )

    try:
        # Build initial observation
        observation = {
            "problem_id": sample["problem_id"],
            "instructions": sample["prompt"],
            "files": list(workspace.files.keys()),
            "index": sample["index"],
            "split": sample["split"],
        }

        # Build messages with STATIC system message and DYNAMIC user message
        # This pattern is required for GEPA pattern mode to work (like Crafter)
        system_message = """You are an expert digital design engineer implementing Verilog spec-to-RTL tasks.

Tools available:
- write_file: Write content to a Verilog file
- compile: Compile sources with iverilog
- simulate: Run simulation with vvp
- submit: Submit solution for grading

Implement the TopModule according to the specification. Use the tools to write your implementation, compile, simulate to verify, and submit when ready."""

        # Dynamic content goes in user message (with wildcards that GEPA can match)
        user_message = f"""Problem: {sample["problem_id"]}

Specification:
{sample["prompt"]}

Available files: {', '.join(workspace.files.keys())}

Please implement the Verilog module. Start by writing the TopModule.v file."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        tools = build_verilog_tools()

        api_key = (
            fastapi_request.headers.get("X-API-Key")
            or fastapi_request.headers.get("x-api-key")
            or None
        )

        http_client = getattr(fastapi_request.app.state, "http_client", None)

        steps: list[RolloutStep] = []
        total_reward = 0.0
        done = False

        # Agentic loop
        for step_idx in range(MAX_STEPS):
            if done:
                break

            # Get model response with tools
            response_text, tool_calls, response_json = await call_chat_completion_with_tools(
                request.policy.config or {},
                messages,
                tools,
                api_key=api_key,
                http_client=http_client,
            )

            # Add assistant message to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response_text or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            step_reward = 0.0
            step_info: dict[str, Any] = {"step": step_idx}

            if not tool_calls:
                # No tool calls - model is done or confused
                step_info["no_tool_call"] = True
                done = True
            else:
                # Process tool calls
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args_str = tc.get("function", {}).get("arguments", "{}")

                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    tool_result: dict[str, Any]

                    if fn_name == TOOL_WRITE_FILE:
                        tool_result = workspace.write_file(
                            fn_args.get("path", "TopModule.v"),
                            fn_args.get("content", ""),
                        )
                    elif fn_name == TOOL_COMPILE:
                        tool_result = workspace.compile(
                            fn_args.get("sources"),
                            fn_args.get("testbench"),
                        )
                    elif fn_name == TOOL_SIMULATE:
                        tool_result = workspace.simulate(fn_args.get("binary"))
                    elif fn_name == TOOL_SUBMIT:
                        tool_result = workspace.submit()
                        done = True
                        if workspace.passed:
                            step_reward = 1.0
                    else:
                        tool_result = {"ok": False, "error": f"Unknown tool: {fn_name}"}

                    step_info[f"tool_{fn_name}"] = tool_result

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(tool_result),
                    })

            total_reward += step_reward

            steps.append(RolloutStep(
                obs=observation,
                tool_calls=tool_calls,
                reward=step_reward,
                done=done,
                info=step_info,
            ))

            print(
                f"[VERILOG_ROLLOUT] seed={seed} step={step_idx} tool_calls={len(tool_calls)} done={done} reward={step_reward}",
                flush=True,
            )

        # Final reward
        final_reward = 1.0 if workspace.passed else 0.0

        trajectory = RolloutTrajectory(
            env_id=f"verilog::{sample['split']}::{sample['index']}",
            policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
            steps=steps,
            final={"observation": observation, "reward": final_reward, "passed": workspace.passed},
            length=len(steps),
            inference_url=str((request.policy.config or {}).get("inference_url") or ""),
        )

        metrics = RolloutMetrics(
            episode_returns=[final_reward],
            mean_return=final_reward,
            num_steps=len(steps),
            num_episodes=1,
            outcome_score=final_reward,
            events_score=final_reward,
            details={"passed": workspace.passed, "steps": len(steps)},
        )

        return RolloutResponse(
            run_id=request.run_id,
            trajectories=[trajectory],
            branches={},
            metrics=metrics,
            aborted=False,
            ops_executed=len(steps),
            trace=None,
        )

    finally:
        # Clean up workspace
        workspace.cleanup()


def build_dataset() -> tuple[TaskDatasetRegistry, VerilogEvalDataset]:
    registry = TaskDatasetRegistry()
    dataset = VerilogEvalDataset()
    registry.register(VERILOG_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "verilog",
            "name": "VerilogEval v2 Spec-to-RTL",
            "version": "2.0.0",
            "action_space": {
                "type": "tool_call",
                "tools": [TOOL_WRITE_FILE, TOOL_COMPILE, TOOL_SIMULATE, TOOL_SUBMIT],
                "description": "Implement Verilog modules using write, compile, simulate, submit workflow.",
            },
        },
        environment="verilog",
        dataset=VERILOG_DATASET_SPEC.model_dump(),
        rubric={"version": "1", "criteria_count": 1, "source": "inline"},
        inference={"supports_proxy": True, "agentic": True},
        limits={"max_turns": MAX_STEPS},
        task_metadata={"format": "agentic_tool_call"},
    )


def describe_taskset(dataset: VerilogEvalDataset) -> Mapping[str, Any]:
    return {
        **VERILOG_DATASET_SPEC.model_dump(),
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: VerilogEvalDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
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
                "problem_id": sample["problem_id"],
                "prompt": sample["prompt"][:500],  # Truncate for metadata
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric({
        "version": "1",
        "goal_text": "Implement Verilog modules that pass testbench verification.",
        "aggregation": "weighted_sum",
        "criteria": [{"id": "testbench_pass", "description": "Implementation passes all testbench tests.", "weight": 1.0}],
    }),
)


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    print("[verilog_task_app] Preloading dataset splits...", flush=True)
    try:
        dataset.ensure_ready(AVAILABLE_SPLITS)
        print(f"[verilog_task_app] Dataset preloaded: {[dataset.size(s) for s in AVAILABLE_SPLITS]}", flush=True)
    except Exception as exc:
        print(f"[verilog_task_app] WARNING: Dataset preload failed: {exc}", flush=True)

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
    )

    async def startup_http_client(app: Any) -> None:
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=60.0)  # Longer timeout for agentic tasks
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            app.state.http_client = aiohttp.ClientSession(timeout=timeout, connector=connector)
            print("[verilog_task_app] Created aiohttp client session", flush=True)
        except Exception as exc:
            print(f"[verilog_task_app] WARNING: Failed to create http client: {exc}", flush=True)
            app.state.http_client = None

    async def shutdown_http_client(app: Any) -> None:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            try:
                await http_client.close()
            except Exception:
                pass

    config = TaskAppConfig(
        app_id="verilog",
        name="VerilogEval v2 Spec-to-RTL Task",
        description="VerilogEval v2 spec-to-RTL task app for Verilog code generation.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=OUTCOME_RUBRIC),
        proxy=proxy_config,
        routers=(verilog_router,),
        app_state={"verilog_dataset": dataset},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="verilog",
        description="VerilogEval v2 spec-to-RTL task app for Verilog code generation.",
        config_factory=build_config,
        aliases=("verilogeval",),
        modal=ModalDeploymentConfig(
            app_name="synth-verilog",
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

    parser = argparse.ArgumentParser(description="Run the VerilogEval task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8118)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
