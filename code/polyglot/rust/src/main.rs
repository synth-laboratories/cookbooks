//! ╔═══════════════════════════════════════════════════════════════════════════╗
//! ║                                                                           ║
//! ║   Synth Task App · Rust                                                   ║
//! ║   Banking77 Intent Classification                                         ║
//! ║                                                                           ║
//! ║   A reference implementation of the Synth Task App contract.              ║
//! ║   This app enables prompt optimization via the GEPA algorithm.            ║
//! ║                                                                           ║
//! ╚═══════════════════════════════════════════════════════════════════════════╝
//!
//! # Architecture
//!
//! This Task App exposes three endpoints that Synth uses during optimization:
//!
//! - `GET  /health`     → Liveness probe (unauthenticated)
//! - `GET  /task_info`  → Describes the task, dataset, and scoring rubric
//! - `POST /rollout`    → Executes one episode: render prompt → call LLM → score
//!
//! The optimization loop works as follows:
//! 1. Synth proposes a candidate prompt template
//! 2. Synth calls `/rollout` with that template + a seed
//! 3. This app renders the prompt, calls the LLM, scores the response
//! 4. Synth uses the score to evolve better prompts
//!
//! # Running
//!
//! ```bash
//! cargo run --release
//! ```
//!
//! # Environment
//!
//! - `ENVIRONMENT_API_KEY` — API key for authenticating Synth requests
//! - `PORT` — Server port (default: 8001)

use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, fs, path::Path, sync::Arc};
use tracing::{info, warn};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A labeled sample from the Banking77 dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sample {
    text: String,
    label: String,
}

/// Dataset loaded from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetFile {
    samples: Vec<Sample>,
    labels: Vec<String>,
}

/// Rollout request from Synth
#[derive(Debug, Deserialize)]
struct RolloutRequest {
    #[serde(default)]
    run_id: String,
    env: EnvSpec,
    policy: PolicySpec,
}

#[derive(Debug, Deserialize)]
struct EnvSpec {
    seed: Option<i64>,
    #[serde(default)]
    config: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct PolicySpec {
    policy_id: Option<String>,
    policy_name: Option<String>,
    #[serde(default)]
    config: HashMap<String, serde_json::Value>,
}

/// Rollout response to Synth
#[derive(Debug, Serialize)]
struct RolloutResponse {
    run_id: String,
    trajectories: Vec<Trajectory>,
    metrics: Metrics,
    aborted: bool,
    ops_executed: i32,
}

#[derive(Debug, Serialize)]
struct Trajectory {
    env_id: String,
    policy_id: String,
    steps: Vec<Step>,
    length: i32,
    inference_url: String,
}

#[derive(Debug, Serialize)]
struct Step {
    obs: HashMap<String, serde_json::Value>,
    tool_calls: Vec<ToolCall>,
    reward: f64,
    done: bool,
    info: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: FunctionCall,
}

#[derive(Debug, Serialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct Metrics {
    episode_returns: Vec<f64>,
    mean_return: f64,
    num_steps: i32,
    num_episodes: i32,
    outcome_score: f64,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Task Info Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Serialize)]
struct TaskInfo {
    task: TaskDescriptor,
    environment: String,
    dataset: DatasetInfo,
    rubric: RubricInfo,
    inference: InferenceInfo,
    limits: LimitsInfo,
}

#[derive(Debug, Serialize)]
struct TaskDescriptor {
    task_id: String,
    name: String,
    description: String,
    version: String,
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    seeds: Vec<i32>,
    train_count: i32,
    val_count: i32,
    test_count: i32,
}

#[derive(Debug, Serialize)]
struct RubricInfo {
    scoring_criteria: String,
    metric_primary: String,
    metric_range: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct InferenceInfo {
    mode: String,
    supported_tools: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LimitsInfo {
    max_response_tokens: i32,
    timeout_seconds: i32,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Configuration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct Config {
    port: u16,
    api_key: Option<String>,
}

impl Config {
    fn from_env() -> Self {
        Self {
            port: env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8001),
            api_key: env::var("ENVIRONMENT_API_KEY").ok(),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Dataset
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct Dataset {
    samples: Vec<Sample>,
    labels: Vec<String>,
}

impl Dataset {
    fn load() -> Self {
        // Try to load from the shared data file
        let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/banking77.json");
        
        if let Ok(content) = fs::read_to_string(&data_path) {
            if let Ok(data) = serde_json::from_str::<DatasetFile>(&content) {
                info!("📊 Loaded {} samples from {:?}", data.samples.len(), data_path);
                return Self {
                    samples: data.samples,
                    labels: data.labels,
                };
            }
        }

        // Fallback to embedded samples
        warn!("⚠️  Dataset not found, using embedded samples");
        let samples = vec![
            Sample { text: "How do I reset my PIN?".into(), label: "change_pin".into() },
            Sample { text: "My card hasn't arrived yet".into(), label: "card_arrival".into() },
            Sample { text: "I want to cancel my card".into(), label: "terminate_account".into() },
            Sample { text: "How do I activate my new card?".into(), label: "activate_my_card".into() },
            Sample { text: "I need to dispute a transaction".into(), label: "transaction_charged_twice".into() },
        ];
        let labels = vec![
            "change_pin".into(), "card_arrival".into(), "terminate_account".into(),
            "activate_my_card".into(), "transaction_charged_twice".into(),
        ];
        Self { samples, labels }
    }

    fn get(&self, index: usize) -> &Sample {
        &self.samples[index % self.samples.len()]
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Prompt Rendering
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Render a template string with placeholder substitution: {key} → value
fn render(template: &str, vars: &HashMap<String, String>) -> String {
    vars.iter().fold(template.to_string(), |acc, (key, value)| {
        acc.replace(&format!("{{{}}}", key), value)
    })
}

/// Build chat messages from a policy config and sample
fn build_messages(
    policy_config: &HashMap<String, serde_json::Value>,
    sample: &Sample,
    labels: &[String],
) -> Vec<ChatMessage> {
    let mut vars = HashMap::new();
    vars.insert("query".into(), sample.text.clone());
    vars.insert("text".into(), sample.text.clone());
    vars.insert("intents".into(), labels.join(", "));

    // Check for prompt_template in policy config
    if let Some(template) = policy_config.get("prompt_template") {
        // Handle object template with sections
        let sections = template
            .get("prompt_sections")
            .or_else(|| template.get("sections"))
            .and_then(|s| s.as_array());

        if let Some(sections) = sections {
            let mut sorted: Vec<_> = sections.iter().collect();
            sorted.sort_by_key(|s| s.get("order").and_then(|o| o.as_i64()).unwrap_or(0));

            return sorted
                .iter()
                .map(|section| {
                    let role = section.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let content = section
                        .get("content")
                        .or_else(|| section.get("pattern"))
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    ChatMessage {
                        role: role.into(),
                        content: render(content, &vars),
                    }
                })
                .collect();
        }
    }

    // Default fallback
    vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a banking assistant. Classify queries using the classify tool.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: format!(
                "Query: {}\nIntents: {}\nClassify this query.",
                sample.text,
                labels.join(", ")
            ),
        },
    ]
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LLM Client
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    tools: Vec<Tool>,
    tool_choice: String,
    temperature: f64,
    max_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ToolFunction,
}

#[derive(Debug, Serialize)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseFunction,
}

#[derive(Debug, Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

/// The classify tool that the LLM must call
fn classify_tool() -> Tool {
    Tool {
        tool_type: "function".into(),
        function: ToolFunction {
            name: "classify".into(),
            description: "Classify the customer query into a banking intent category".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "The classified intent"
                    }
                },
                "required": ["intent"]
            }),
        },
    }
}

/// Call an OpenAI-compatible LLM endpoint (Synth provides authenticated inference_url)
async fn call_llm(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
) -> Result<ChatResponse> {
    // Construct the chat completions URL, preserving any query params
    let url = if let Some(q_idx) = base_url.find('?') {
        let (base, query) = base_url.split_at(q_idx);
        format!("{}/chat/completions{}", base.trim_end_matches('/'), query)
    } else {
        format!("{}/chat/completions", base_url.trim_end_matches('/'))
    };

    let request = ChatRequest {
        model: model.into(),
        messages,
        tools: vec![classify_tool()],
        tool_choice: "required".into(),
        temperature: 0.0,
        max_tokens: 100,
    };

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("LLM error: {} {}", status, body);
    }

    Ok(response.json().await?)
}

/// Extract prediction and tool calls from LLM response
fn extract_prediction(response: &ChatResponse) -> (Option<String>, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut prediction = None;

    if let Some(choice) = response.choices.first() {
        // Extract from tool calls
        if let Some(calls) = &choice.message.tool_calls {
            for call in calls {
                tool_calls.push(ToolCall {
                    id: call.id.clone(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone(),
                    },
                });

                if call.function.name == "classify" {
                    if let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
                        prediction = args.get("intent").and_then(|i| i.as_str()).map(String::from);
                    }
                }
            }
        }

        // Fallback to raw content if no tool call
        if prediction.is_none() {
            prediction = choice.message.content.as_ref().map(|c| c.trim().to_string());
        }
    }

    (prediction, tool_calls)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// App State
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct AppState {
    config: Config,
    dataset: Dataset,
    http_client: reqwest::Client,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HTTP Handlers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Serialize)]
struct HealthResponse {
    healthy: bool,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    code: String,
    message: String,
}

/// Verify API key for protected routes
fn require_auth(state: &AppState, headers: &HeaderMap) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(expected) = &state.config.api_key {
        let provided = headers.get("x-api-key").and_then(|v| v.to_str().ok());
        if provided != Some(expected.as_str()) {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        code: "unauthorised".into(),
                        message: "API key missing or invalid".into(),
                    },
                }),
            ));
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /health — Liveness probe
// ─────────────────────────────────────────────────────────────────────────────
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { healthy: true })
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /task_info — Task metadata and available seeds
// ─────────────────────────────────────────────────────────────────────────────
async fn task_info_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<Vec<TaskInfo>>, (StatusCode, Json<ErrorResponse>)> {
    require_auth(&state, &headers)?;

    // Parse requested seeds from query string
    let query_string = req.uri().query().unwrap_or("");
    let requested: Vec<i32> = query_string
        .split('&')
        .filter_map(|param| {
            let mut parts = param.split('=');
            match (parts.next(), parts.next()) {
                (Some("seed"), Some(val)) | (Some("seeds"), Some(val)) => val.parse().ok(),
                _ => None,
            }
        })
        .collect();

    let all_seeds: Vec<i32> = (0..state.dataset.len() as i32).collect();
    let seed_groups = if requested.is_empty() {
        vec![all_seeds.clone()]
    } else {
        requested.iter().map(|s| vec![*s]).collect()
    };

    let infos = seed_groups
        .iter()
        .map(|seeds| TaskInfo {
            task: TaskDescriptor {
                task_id: "banking77-rust".into(),
                name: "Banking77 Intent Classification".into(),
                description: "Classify banking customer queries into intent categories".into(),
                version: "1.0.0".into(),
            },
            environment: "banking77".into(),
            dataset: DatasetInfo {
                seeds: seeds.clone(),
                train_count: state.dataset.len() as i32,
                val_count: 0,
                test_count: 0,
            },
            rubric: RubricInfo {
                scoring_criteria: "exact_match".into(),
                metric_primary: "accuracy".into(),
                metric_range: vec![0.0, 1.0],
            },
            inference: InferenceInfo {
                mode: "tool_call".into(),
                supported_tools: vec!["classify".into()],
            },
            limits: LimitsInfo {
                max_response_tokens: 100,
                timeout_seconds: 30,
            },
        })
        .collect();

    Ok(Json(infos))
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /rollout — Execute one classification episode
// ─────────────────────────────────────────────────────────────────────────────
async fn rollout_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RolloutRequest>,
) -> Result<Json<RolloutResponse>, (StatusCode, Json<ErrorResponse>)> {
    require_auth(&state, &headers)?;

    let seed = req.env.seed.unwrap_or(0) as usize;
    let sample = state.dataset.get(seed);

    // Resolve inference URL
    let inference_url = req.policy.config
        .get("inference_url")
        .or_else(|| req.policy.config.get("api_base"))
        .or_else(|| req.policy.config.get("base_url"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        code: "bad_request".into(),
                        message: "Missing inference_url in policy.config".into(),
                    },
                }),
            )
        })?;

    let model = req.policy.config
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4o-mini");

    let messages = build_messages(&req.policy.config, sample, &state.dataset.labels);

    // Call LLM
    let llm_response = call_llm(&state.http_client, inference_url, model, messages)
        .await
        .map_err(|e| {
            warn!("❌ LLM call failed: {}", e);
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        code: "llm_error".into(),
                        message: format!("LLM call failed: {}", e),
                    },
                }),
            )
        })?;

    let (prediction, tool_calls) = extract_prediction(&llm_response);

    // Score: exact match on intent label
    let correct = prediction
        .as_ref()
        .map(|p| p.to_lowercase() == sample.label.to_lowercase())
        .unwrap_or(false);
    let reward = if correct { 1.0 } else { 0.0 };

    info!(
        "🎯 seed={} expected=\"{}\" predicted=\"{:?}\" {}",
        seed,
        sample.label,
        prediction,
        if correct { "✓" } else { "✗" }
    );

    // Build response
    let mut obs = HashMap::new();
    obs.insert("query".into(), serde_json::json!(sample.text));
    obs.insert("index".into(), serde_json::json!(seed));

    let mut info = HashMap::new();
    info.insert("expected".into(), serde_json::json!(sample.label));
    info.insert("predicted".into(), serde_json::json!(prediction));
    info.insert("correct".into(), serde_json::json!(correct));

    let response = RolloutResponse {
        run_id: req.run_id,
        trajectories: vec![Trajectory {
            env_id: format!("task::train::{}", seed),
            policy_id: req.policy.policy_id.or(req.policy.policy_name).unwrap_or_else(|| "policy".into()),
            inference_url: inference_url.into(),
            length: 1,
            steps: vec![Step {
                obs,
                tool_calls,
                reward,
                done: true,
                info,
            }],
        }],
        metrics: Metrics {
            episode_returns: vec![reward],
            mean_return: reward,
            num_steps: 1,
            num_episodes: 1,
            outcome_score: reward,
        },
        aborted: false,
        ops_executed: 1,
    };

    Ok(Json(response))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Server
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env();
    let dataset = Dataset::load();

    println!(
        r#"
╭───────────────────────────────────────╮
│  Synth Task App · Banking77           │
│  Port: {:<5}                          │
│  Auth: {}                        │
╰───────────────────────────────────────╯
"#,
        config.port,
        if config.api_key.is_some() { "enabled ✓" } else { "disabled ⚠" }
    );

    let state = Arc::new(AppState {
        config,
        dataset,
        http_client: reqwest::Client::new(),
    });

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/task_info", get(task_info_handler))
        .route("/rollout", post(rollout_handler))
        .with_state(state.clone());

    let addr = format!("0.0.0.0:{}", state.config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
