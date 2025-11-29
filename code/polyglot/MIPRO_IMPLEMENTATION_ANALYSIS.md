# MIPRO Implementation Analysis - OpenAPI Design Reference

**Generated**: 2025-11-22  
**Codebase**: monorepo (monorepo/backend/app/routes/prompt_learning/algorithm/mipro)  
**Purpose**: Complete guide for designing REST/OpenAPI interfaces for MIPRO prompt optimization

---

## 1. MIPRO CORE COMPONENTS

### 1.1 Main Optimizer Class
**Location**: `monorepo/backend/app/routes/prompt_learning/algorithm/mipro/optimizer/optimizer.py`

```python
class MIPROOptimizer:
    """Main MIPROv2 optimization loop."""
    
    def __init__(
        self,
        job_id: str,
        config: MIPROConfig,
        emitter: Optional[EventEmitterProtocol] = None,
        runtime: Optional[OptimizerRuntime] = None,
        http_client=None,
        initial_prompt_config: Optional[Dict[str, Any]] = None,
    )
    
    async def optimize(self) -> OptimizationResult:
        """Main optimization loop - coordinates bootstrap, proposal, and evaluation."""
```

**Key Entry Point**: `optimize()` - Asynchronous method that orchestrates entire MIPRO workflow.

---

## 2. DATA STRUCTURES & TYPES

### 2.1 Configuration Types
**Location**: `monorepo/backend/app/routes/prompt_learning/algorithm/mipro/config.py`

#### MIPROConfig (Main)
```python
@dataclass
class MIPROConfig:
    task_app_url: str              # URL to task application
    task_app_api_key: str          # API key for task app auth
    seeds: MIPROSeedConfig         # Train/online/test/validation seeds
    env_name: str                  # Environment name (e.g., "banking77")
    policy_config: Dict[str, Any]  # Model/provider config for policy
    env_config: Optional[Dict[str, Any]]  # Environment-specific config
    
    # Optimization parameters
    num_iterations: int = 2
    num_evaluations_per_iteration: int = 3
    batch_size: int = 5
    max_concurrent: int = 10
    parallel_risk: str = "LOW"     # Safety knob for parallelization
    max_parallel_trials: int = 1
    
    # Demo and instruction limits
    max_demo_set_size: int = 4
    max_demo_sets: int = 128
    max_instruction_sets: int = 128
    few_shot_score_threshold: float = 0.8
    
    # Meta-model for proposal generation
    meta: MIPROMetaConfig
    
    # Instruction configuration
    instructions: MIPROInstructionConfig
    
    # Module and stage configuration (multi-stage pipelines)
    modules: List[MIPROModuleConfig] = []
    stages: List[MIPROStageConfig] = []
    
    # Optional features
    spec: Optional[Dict[str, Any]] = None        # System spec for synth proposer
    metaprompt: Optional[str] = None             # Custom meta-prompt
    judge: Optional[JudgeConfig] = None          # Judge config for scoring
    proxy_models: Optional[ProxyModelsConfig] = None  # Proxy model config
    adaptive_pool: Optional[AdaptivePoolConfig] = None
    tpe: Optional[MIPROTPEConfig] = None         # Tree-structured Parzen Estimator config
    
    # Cost and termination
    max_token_limit: Optional[int] = None
    max_spend_usd: Optional[float] = None
    max_rollout_spend_usd: Optional[float] = None
    max_proposal_spend_usd: Optional[float] = None
    termination_conditions: Optional[TerminationConditions] = None
    termination_config: Optional[TerminationConfig] = None
```

#### MIPROSeedConfig
```python
@dataclass
class MIPROSeedConfig:
    bootstrap: List[int]      # Seeds for collecting correct examples (~10-20)
    online: List[int]         # Seeds for mini-batch eval during optimization (~20-50)
    test: List[int] = []      # Held-out seeds for final eval
    reference: List[int] = [] # Seeds for reference corpus in meta-prompt (validation)
```

#### MIPROMetaConfig
```python
@dataclass
class MIPROMetaConfig:
    model: str = "openai/gpt-oss-120b"  # Meta-model for instruction proposal
    provider: str = "groq"              # Model provider
    inference_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 600
```

#### MIPROInstructionConfig
```python
@dataclass
class MIPROInstructionConfig:
    instructions_per_batch: int = 10
    max_instructions: int = 1                          # Max instructions per stage
    duplicate_retry_limit: int = 10
    generate_at_iterations: List[int] = [0]           # When to generate new instructions
    proposer_mode: Literal["dspy", "synth", "gepa-ai"] = "synth"
    proposer_effort: Literal["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"] = "LOW"
    proposer_output_tokens: Literal["RAPID", "FAST", "SLOW"] = "FAST"
    transform_strict: bool = False
    transform_case_sensitive: bool = True
    transform_match_mode: Literal["exact", "substring"] = "substring"
```

#### MIPROStageConfig (Multi-stage pipeline)
```python
@dataclass
class MIPROStageConfig:
    stage_id: str
    max_instruction_slots: int = 1
    max_demo_slots: int = 0
    baseline_instruction: Optional[str] = None
    baseline_messages: List[Dict[str, Any]] = []
    policy_config: Dict[str, Any] = {}  # Per-stage policy
```

#### MIPROModuleConfig (Multi-stage pipeline)
```python
@dataclass
class MIPROModuleConfig:
    module_id: str
    stages: List[MIPROStageConfig] = []
    edges: List[Tuple[str, str]] = []  # Directed edges between stages
```

---

### 2.2 Core Type Definitions
**Location**: `monorepo/backend/app/routes/prompt_learning/types/mipro_types.py`

#### Proposal
```python
@dataclass
class Proposal:
    instruction_text: str         # Combined instruction text for evaluation
    demo_indices: list[int]       # Indices of demos to include
    proposal_id: str              # Unique proposal ID
    instruction_indices: list[int] = []
    stage_payloads: Dict[str, StageProposalPayload] = {}  # Per-stage data
```

#### Candidate
```python
@dataclass
class Candidate:
    trial_num: int                # Trial number (0-indexed)
    instruction_text: str         # Combined instruction text
    demo_indices: list[int]       # Selected demo indices
    version_id: str               # PromptTransformation version ID
    instruction_indices: list[int] = []
    instruction_lines: list[str] = []
    
    # Evaluation results
    minibatch_score: float | None = None  # Score on minibatch
    full_score: float | None = None       # Score on full validation set
    
    # Metadata
    is_incumbent: bool = False
    iteration: int = 0
    stage_payloads: Dict[str, StageCandidatePayload] = {}
```

#### FewShotExample
```python
@dataclass
class FewShotExample:
    seed: int                 # Seed this example was generated from
    input_data: dict[str, Any]   # Task input
    output_data: dict[str, Any]  # Task output
    score: float              # Task score (0.0-1.0)
    messages: list[dict[str, Any]]  # Formatted messages for few-shot prompt
```

#### PromptInstruction
```python
@dataclass
class PromptInstruction:
    instruction_id: int
    text: str
    score: float = float("-inf")
    demo_indices: list[int] = []
    num_trials: int = 0
    total_score: float = 0.0
    stage_id: str = "stage_default"
    module_id: str = "module_default"
```

---

### 2.3 Optimization Result
```python
@dataclass(slots=True)
class OptimizationResult:
    best_candidate: Candidate
    best_minibatch_score: float
    best_full_score: Optional[float]
    test_score: Optional[float]
    total_trials: int
    
    # Cost and balance
    total_cost_usd: float = 0.0
    category_costs: Dict[str, float] = {}
    final_balance_usd: Optional[float] = None
    balance_type: Optional[str] = None
    
    # Candidate tracking
    attempted_candidates: List[Dict[str, Any]] = []  # All candidates
    optimized_candidates: List[Dict[str, Any]] = [] # Improved candidates
```

### 2.4 Bootstrap Result
```python
@dataclass(slots=True)
class BootstrapResult:
    baseline_score: float         # Score of baseline prompt
    few_shot_examples: Dict[str, List[FewShotExample]]  # By stage_id
```

---

## 3. MIPRO WORKFLOW & DATA FLOW

### 3.1 High-Level Optimization Flow

```
MIPROOptimizer.optimize()
│
├─ 1. Health Check Task App
│
├─ 2. Validate Seeds Exist
│
├─ 3. Start Interceptor (proxies LLM calls)
│
├─ 4. Gather Baseline Messages (for multi-stage)
│
├─ 5. BOOTSTRAP PHASE
│   ├─ Evaluate baseline prompt on bootstrap seeds
│   └─ Collect correct examples (few-shot pool)
│
├─ OPTIMIZATION LOOP (iterations 0..N)
│   │
│   ├─ FOR EACH ITERATION:
│   │   │
│   │   ├─ PROPOSAL GENERATION (if configured for this iteration)
│   │   │   ├─ Call proposer (synth/dspy/gepa-ai)
│   │   │   ├─ Generate atomic instruction transforms
│   │   │   └─ Add to transform bank
│   │   │
│   │   ├─ BUILD SEARCH SPACE
│   │   │   └─ Create discrete dimensions for instructions × demos
│   │   │
│   │   ├─ TRIAL EVALUATION BATCH
│   │   │   ├─ TPE suggests top-K candidates
│   │   │   ├─ FOR EACH CANDIDATE:
│   │   │   │   ├─ Compile instruction + demos
│   │   │   │   ├─ Register with interceptor
│   │   │   │   ├─ Evaluate on online pool (minibatch)
│   │   │   │   └─ Store result (score + cost)
│   │   │   │
│   │   │   ├─ FULL EVALUATION (every K iterations)
│   │   │   │   └─ Evaluate best candidate on full validation set
│   │   │   │
│   │   │   └─ TPE update with new observations
│   │   │
│   │   └─ CHECK TERMINATION CONDITIONS
│
└─ 6. FINAL EVALUATION
    ├─ Evaluate best candidate on test pool
    └─ Return OptimizationResult
```

---

## 4. PROPOSER INTERFACES

### 4.1 Proposer Classes
**Location**: `monorepo/backend/app/routes/prompt_learning/algorithm/mipro/proposers/`

#### Abstract Base: InstructionProposer
```python
class InstructionProposer:
    async def propose_transforms(
        self,
        ctx: StageInstructionContext,
        history: OptimizerHistory,
        dataset: OptimizerStartingDataset,
    ) -> TransformProposalBatch:
        """Generate instruction transforms for a stage."""
```

#### Available Proposers
1. **SynthMIPROInstructionProposer** - Synth-branded variant (recommended)
2. **DSPyMIPROInstructionProposer** - DSPy-based proposer
3. **BuiltinMIPROInstructionProposer** - Built-in basic proposer

### 4.2 Proposer Modes
```
proposer_mode: Literal["dspy", "synth", "gepa-ai"]
```

- **"synth"**: Uses Synth Research templates + spec context
- **"dspy"**: Original DSPy-based implementation
- **"gepa-ai"**: Maps to synth (alias for backwards compatibility)

### 4.3 Proposer Effort Levels
```
proposer_effort: Literal["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"]
```

Controls meta-model selection (model, provider, temperature, tokens)

---

## 5. INSTRUCTION GENERATION & TRANSFORMATION

### 5.1 Atomic Instruction Transform
```python
@dataclass
class AtomicInstructionTransform:
    """Single instruction modification that can be composed."""
    
    instruction_type: InstructionType  # REPLACE, FOLLOW, PYTHON
    localizer: InstructionTransformLocalizer  # How to locate in prompt
    instruction_text: str  # What to insert/replace with
    demo_indices: list[int]  # Relevant few-shot examples
```

### 5.2 Transform Compilation
**Location**: `monorepo/backend/app/routes/prompt_learning/algorithm/mipro/transform_compiler.py`

```python
async def compile_prompt_from_transforms(
    base_messages: List[Dict[str, str]],
    transforms: List[AtomicInstructionTransform],
    demo_pool: List[FewShotExample],
) -> Tuple[List[Dict[str, str]], int]:
    """Apply transforms and demo injection to create final prompt."""
    # 1. Apply instruction transforms (replace/follow/python)
    # 2. Inject few-shot examples at appropriate positions
    # 3. Return final messages + token count
```

---

## 6. TPE OPTIMIZATION

### 6.1 Tree-structured Parzen Estimator (TPE)
**Location**: `monorepo/backend/app/routes/prompt_learning/algorithm/mipro/tpe.py`

```python
class AsyncTPEOptimizer:
    """Async TPE for instruction × demo set optimization."""
    
    async def suggest_ranked(self, top_k: int) -> List[Dict[str, int]]:
        """Return top_k ranked configurations by Expected Improvement."""
    
    async def observe(self, config: Dict[str, int], score: float):
        """Record trial observation for learning."""
```

### 6.2 TPE Configuration
```python
@dataclass
class MIPROTPEConfig:
    gamma: Optional[float] = None           # Quantile (0-1, default 0.25)
    n_candidates: Optional[int] = None      # Candidates to eval EI (default 24)
    n_startup_trials: Optional[int] = None  # Random trials (default 10)
    epsilon: Optional[float] = None         # Exploration prob (0-1, default 0.25)
    alpha: Optional[float] = None           # KDE smoothing (default 1.0)
```

---

## 7. REST API ENDPOINTS (EXISTING)

**Location**: `monorepo/backend/app/routes/prompt_learning/routes_online.py`

### 7.1 Current Endpoints

#### POST /api/prompt-learning/online/jobs
Create and start a prompt learning job.

**Request**:
```json
{
  "algorithm": "mipro",
  "config_name": "optional_saved_config",
  "config_body": {
    "prompt_learning": {
      "algorithm": "mipro",
      "task_app_url": "http://task-app:8102",
      "task_app_id": "banking77"
      // ... MIPROConfig fields
    }
  },
  "overrides": {},
  "metadata": {},
  "auto_start": true
}
```

**Response** (200):
```json
{
  "job_id": "job_abc123",
  "algorithm": "mipro",
  "status": "running",
  "created_at": "2025-11-12T10:00:00Z",
  "started_at": "2025-11-12T10:00:01Z",
  "finished_at": null,
  "best_score": null
}
```

#### GET /api/prompt-learning/online/jobs/{job_id}
Get job details.

#### GET /api/prompt-learning/online/jobs/{job_id}/events
Get job events/logs.

#### GET /api/prompt-learning/online/jobs/{job_id}/events/stream
Stream job events (SSE).

#### GET /api/prompt-learning/online/jobs/{job_id}/metrics
Get job metrics (scores, costs).

---

## 8. CONFIGURATION PARSING

**Location**: `monorepo/backend/app/routes/prompt_learning/algorithm/mipro/config.py`

```python
def parse_mipro_config(
    pl_config: Mapping[str, Any],
    *,
    task_app_url: str,
    task_app_api_key: str,
) -> MIPROConfig:
    """Parse prompt-learning TOML section into MIPROConfig."""
```

### 8.1 TOML Config Example
```toml
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://127.0.0.1:8102"
task_app_id = "banking77"
bootstrap_train_seeds = [0, 1, 2, ..., 14]
online_pool = [20, 21, ..., 39]
test_pool = [40, 41, ..., 89]
val_seeds = [100, 101, ..., 109]  # For validation evaluation

[prompt_learning.policy]
model = "openai/gpt-oss-20b"
provider = "groq"
temperature = 0.0
max_completion_tokens = 512

[prompt_learning.mipro]
num_iterations = 2
num_evaluations_per_iteration = 3
batch_size = 5
meta_model = "openai/gpt-oss-120b"
meta_model_provider = "groq"
few_shot_score_threshold = 0.8

[prompt_learning.mipro.tpe]
gamma = 0.25
n_candidates = 32
n_startup_trials = 10
```

---

## 9. KEY CLASSES & FILES SUMMARY

| Component | Location | Purpose |
|-----------|----------|---------|
| **MIPROOptimizer** | optimizer/optimizer.py | Main optimization loop |
| **MIPROConfig** | config.py | Configuration dataclass |
| **MIPROMetaConfig** | config.py | Meta-model settings |
| **MIPROInstructionConfig** | config.py | Instruction generation settings |
| **Proposal** | types/mipro_types.py | Proposal structure |
| **Candidate** | types/mipro_types.py | Evaluated candidate |
| **FewShotExample** | types/mipro_types.py | Few-shot demo |
| **OptimizationResult** | optimizer/optimizer.py | Final result |
| **SynthMIPROInstructionProposer** | proposers/synth_proposer.py | Instruction proposer (synth mode) |
| **DSPyMIPROInstructionProposer** | proposers/dspy_proposer.py | Instruction proposer (dspy mode) |
| **AsyncTPEOptimizer** | tpe.py | TPE optimizer |
| **TransformCompiler** | transform_compiler.py | Prompt compilation |
| **InferenceInterceptor** | core/inference_interceptor.py | LLM call proxy |

---

## 10. KEY ALGORITHMS & PATTERNS

### 10.1 Bootstrapping
1. Evaluate baseline prompt on bootstrap seeds
2. Collect correct examples (score >= threshold)
3. Store as few-shot pool for later injection

### 10.2 Instruction Proposal
1. Meta-model reads:
   - Task description
   - Few-shot examples
   - Current program code (if available)
   - Previous successful instructions
2. Generates N atomic instructions
3. Each instruction is a transform (REPLACE, FOLLOW, PYTHON)
4. Deduplicate and add to transform bank

### 10.3 Candidate Generation (TPE)
1. Discrete search space: instruction_slot_1, instruction_slot_2, demo_slot_1, etc.
2. Each dimension has options from transform bank
3. TPE learns good (l) and bad (g) distributions
4. Suggests top-K combinations by Expected Improvement (EI = l/g)
5. Epsilon-greedy: explore with probability ε

### 10.4 Evaluation
1. Compile prompt: apply transforms + inject demos
2. Register versioned prompt with interceptor
3. Task app calls policy model via interceptor URL
4. Collect results: score + token usage + cost
5. Update TPE with observation

---

## 11. INPUTS & OUTPUTS SPECIFICATION

### 11.1 Inputs to optimize()

**From MIPROConfig**:
- `task_app_url`: Task app endpoint
- `task_app_api_key`: Authentication
- `bootstrap_seeds`: Seeds for few-shot collection
- `online_seeds`: Seeds for minibatch evaluation
- `test_seeds`: Held-out test set
- `val_seeds`: Seeds for full evaluation
- `policy_config`: Model/provider for policy
- `num_iterations`: Optimization iterations
- `num_evaluations_per_iteration`: Trials per iteration
- `meta`: Meta-model config
- `instructions`: Proposer config
- `modules`: Multi-stage pipeline config

**From initial_prompt_config**:
- `messages`: Initial prompt messages
- `wildcards`: Template variables

### 11.2 Outputs from optimize()

**OptimizationResult**:
- `best_candidate`: Best Candidate found
  - `instruction_text`: Optimized instructions
  - `demo_indices`: Selected few-shot examples
  - `minibatch_score`: Score on online pool
  - `full_score`: Score on validation set (if full eval)
- `best_minibatch_score`: Best minibatch score achieved
- `best_full_score`: Best validation score (if full eval)
- `test_score`: Score on held-out test set
- `total_trials`: Total candidates evaluated
- `total_cost_usd`: Total optimization cost
- `category_costs`: Cost breakdown by category
- `final_balance_usd`: Remaining user balance

---

## 12. ERROR HANDLING & TERMINATION

### 12.1 Error Conditions
- **MIPROConfigurationError**: Invalid config
- **BudgetExceededError**: Token/spend budget exceeded
- **TransformApplicationError**: Transform compilation failed
- **TaskAppHealthCheckFailed**: Task app not available

### 12.2 Termination Conditions
```python
@dataclass
class TerminationConditions:
    max_error_rate: Optional[float] = None      # % of trials can fail
    max_consecutive_errors: Optional[int] = None # Consecutive failures
    # ... other conditions
```

### 12.3 Termination Config
```python
@dataclass
class TerminationConfig:
    max_duration_minutes: Optional[float] = None
    max_total_spend_usd: Optional[float] = None
    max_trials_allowed: Optional[int] = None
```

---

## 13. TASK APP INTERFACE

### 13.1 Task App Contract
**Health Check**: `GET /health` → 200 OK

**Task Info**: `POST /task_info` → Task metadata

**Rollout**: `POST /rollout`
```json
{
  "policy_url": "https://interceptor.example.com/v1/{version_id}/chat/completions",
  "seeds": [20, 21, 22, ...],
  "policy_config": {...}
}
```

Returns: Score + token usage per seed

---

## 14. DESIGN RECOMMENDATIONS FOR OPENAPI SPEC

### 14.1 Core Endpoint Categories
1. **Job Management** (create, list, get, start, cancel)
2. **Configuration** (get initial config, override)
3. **Execution Monitoring** (events stream, metrics)
4. **Results & Artifacts** (best prompt, snapshots, history)

### 14.2 Request/Response Schemas
- Use `MIPROConfig` schema for config body
- Use `OptimizationResult` schema for job completion response
- Use `Candidate` schema for best prompt
- Use event streaming for real-time progress

### 14.3 Authentication
- Bearer token in Authorization header
- API key identifies org for billing

### 14.4 Common Query Parameters
- `limit`: Pagination
- `offset`: Pagination offset
- `since`: Filter by timestamp

---

## 15. KEY FILES TO REFERENCE

```
monorepo/backend/app/routes/prompt_learning/
├── algorithm/mipro/
│   ├── optimizer/optimizer.py         # Main optimizer
│   ├── config.py                      # MIPROConfig definition + parsing
│   ├── mipro_config.py               # Constants & defaults
│   ├── miprov2_config.py             # MIPROv2-specific defaults
│   ├── tpe.py                        # TPE optimizer
│   ├── proposers/
│   │   ├── synth_proposer.py         # Synth proposer
│   │   ├── dspy_proposer.py          # DSPy proposer
│   │   ├── builtin.py                # Base proposer
│   │   └── abstractions.py           # Transform types
│   ├── transform_compiler.py         # Prompt compilation
│   ├── parallel_batch_planner.py     # Parallel evaluation
│   └── parallel_risk.py              # Risk settings
├── types/mipro_types.py              # Core types (Proposal, Candidate, etc.)
├── core/
│   ├── config.py                     # Config loading
│   ├── evaluation.py                 # Evaluation helpers
│   ├── inference_interceptor.py      # LLM proxy
│   ├── task_app_client.py            # Task app API client
│   └── prompt.py                     # Prompt structures
├── routes_online.py                  # REST API endpoints
└── configs/*.toml                    # Example configs
```

---

## SUMMARY FOR OPENAPI DESIGN

**Core Business Logic**:
- MIPRO is a Bayesian optimization algorithm for prompt improvement
- Inputs: task app, policy model, seed examples
- Process: bootstrap → propose → evaluate → optimize
- Output: best prompt + score

**Key Data Structures**:
- `MIPROConfig`: Configuration
- `Candidate`: Evaluated prompt
- `OptimizationResult`: Final result
- `FewShotExample`: Demonstration

**REST Integration Points**:
- Job CRUD (create, list, get, start)
- Event streaming (SSE)
- Metrics querying
- Artifact management

**Critical Async Operations**:
- `optimize()` - Main async orchestrator
- `propose_transforms()` - Proposal generation
- Parallel seed evaluation
- TPE suggestion ranking

---

