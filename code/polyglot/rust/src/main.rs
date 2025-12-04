//! ╔═══════════════════════════════════════════════════════════════════════════╗
//! ║                                                                           ║
//! ║   Synth Task App · Rust                                                   ║
//! ║   Banking77 Intent Classification with Integrated Smoke Test              ║
//! ║                                                                           ║
//! ║   A reference implementation of the Synth Task App contract.              ║
//! ║   This app enables prompt optimization via the GEPA algorithm.            ║
//! ║                                                                           ║
//! ╚═══════════════════════════════════════════════════════════════════════════╝
//!
//! # Running as Task App Server
//!
//! ```bash
//! cargo run --release
//! ```
//!
//! # Running End-to-End Smoke Test
//!
//! ```bash
//! cargo run --release -- --smoke-test
//! ```
//!
//! This will:
//! 1. Start the task app server
//! 2. Create a Cloudflare tunnel
//! 3. Submit a GEPA job to production
//! 4. Stream events until completion
//!
//! # Environment
//!
//! - `ENVIRONMENT_API_KEY` — API key for authenticating Synth requests
//! - `SYNTH_API_KEY` — API key for backend (smoke test only)
//! - `PORT` — Server port (default: 8001)

use anyhow::{anyhow, Result};
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, fs, path::Path, process::Stdio, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    time::sleep,
};
use tracing::{info, warn};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sample {
    text: String,
    label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetFile {
    samples: Vec<Sample>,
    labels: Vec<String>,
}

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
    synth_api_key: Option<String>,
    backend_url: String,
}

impl Config {
    fn from_env() -> Self {
        // Try to load from synth-ai/.env
        let env_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../synth-ai/.env");
        if env_path.exists() {
            if let Ok(content) = fs::read_to_string(&env_path) {
                for line in content.lines() {
                    let line = line.trim();
                    if !line.is_empty() && !line.starts_with('#') {
                        if let Some((key, value)) = line.split_once('=') {
                            let value = value.trim_matches(|c| c == '"' || c == '\'');
                            if env::var(key).is_err() {
                                env::set_var(key, value);
                            }
                        }
                    }
                }
                println!("✓ Loaded .env from synth-ai");
            }
        }

        Self {
            port: env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8001),
            api_key: env::var("ENVIRONMENT_API_KEY").ok(),
            synth_api_key: env::var("SYNTH_API_KEY").ok(),
            backend_url: env::var("BACKEND_URL")
                .unwrap_or_else(|_| "https://agent-learning.onrender.com/api".into()),
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
        let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/banking77.json");

        if let Ok(content) = fs::read_to_string(&data_path) {
            if let Ok(data) = serde_json::from_str::<DatasetFile>(&content) {
                println!("📊 Loaded {} samples", data.samples.len());
                return Self {
                    samples: data.samples,
                    labels: data.labels,
                };
            }
        }

        println!("⚠️  Dataset not found, using embedded samples");
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

fn render(template: &str, vars: &HashMap<String, String>) -> String {
    vars.iter().fold(template.to_string(), |acc, (key, value)| {
        acc.replace(&format!("{{{}}}", key), value)
    })
}

fn build_messages(
    policy_config: &HashMap<String, serde_json::Value>,
    sample: &Sample,
    labels: &[String],
) -> Vec<ChatMessage> {
    let mut vars = HashMap::new();
    vars.insert("query".into(), sample.text.clone());
    vars.insert("text".into(), sample.text.clone());
    vars.insert("intents".into(), labels.join(", "));

    if let Some(template) = policy_config.get("prompt_template") {
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

async fn call_llm(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
) -> Result<ChatResponse> {
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

fn extract_prediction(response: &ChatResponse) -> (Option<String>, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut prediction = None;

    if let Some(choice) = response.choices.first() {
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

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { healthy: true })
}

async fn task_info_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Result<Json<Vec<TaskInfo>>, (StatusCode, Json<ErrorResponse>)> {
    require_auth(&state, &headers)?;

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

async fn rollout_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RolloutRequest>,
) -> Result<Json<RolloutResponse>, (StatusCode, Json<ErrorResponse>)> {
    require_auth(&state, &headers)?;

    let seed = req.env.seed.unwrap_or(0) as usize;
    let sample = state.dataset.get(seed);

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
// Cloudflare Tunnel
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async fn create_tunnel(port: u16) -> Result<(String, tokio::process::Child)> {
    println!("🌐 Creating Cloudflare tunnel for port {}...", port);

    let mut child = Command::new("cloudflared")
        .args(["tunnel", "--config", "/dev/null", "--url", &format!("http://127.0.0.1:{}", port)])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stderr = child.stderr.take().ok_or_else(|| anyhow!("No stderr"))?;
    let mut reader = BufReader::new(stderr).lines();

    // Wait for tunnel URL (up to 30 seconds)
    let url = tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(line) = reader.next_line().await? {
            if let Some(start) = line.find("https://") {
                if let Some(end) = line[start..].find(".trycloudflare.com") {
                    let url = &line[start..start + end + ".trycloudflare.com".len()];
                    return Ok::<String, anyhow::Error>(url.to_string());
                }
            }
        }
        Err(anyhow!("Tunnel URL not found in output"))
    })
    .await
    .map_err(|_| anyhow!("Timeout waiting for tunnel"))??;

    println!("✅ Tunnel created: {}", url);

    // Wait a bit for DNS propagation
    println!("⏳ Waiting for tunnel DNS (5s)...");
    sleep(Duration::from_secs(5)).await;

    Ok((url, child))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Smoke Test Job Submission & Polling
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Deserialize)]
struct JobResponse {
    job_id: Option<String>,
    id: Option<String>,
    status: Option<String>,
    error: Option<String>,
    metadata: Option<serde_json::Value>,
}

async fn submit_job(
    client: &reqwest::Client,
    backend_url: &str,
    synth_api_key: &str,
    task_app_url: &str,
    task_app_api_key: &str,
) -> Result<String> {
    println!("📤 Submitting GEPA job...");
    println!("   Task App URL: {}", task_app_url);

    let config_body = serde_json::json!({
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": task_app_url,
            "task_app_api_key": task_app_api_key,
            "initial_prompt": {
                "id": "banking77_rust",
                "name": "Banking77 Classification (Rust)",
                "messages": [
                    {"role": "system", "pattern": "You are an expert banking assistant. Classify queries using the classify tool.", "order": 0},
                    {"role": "user", "pattern": "Query: {query}\nIntents: {intents}\nClassify this query.", "order": 1}
                ],
                "wildcards": {"query": "REQUIRED", "intents": "REQUIRED"}
            },
            "policy": {
                "inference_mode": "synth_hosted",
                "model": "gpt-4.1-nano",
                "provider": "openai",
                "temperature": 0.0,
                "max_completion_tokens": 64
            },
            "gepa": {
                "env_name": "banking77",
                "proposer_effort": "MEDIUM",
                "proposer_output_tokens": "FAST",
                "evaluation": {
                    "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    "validation_seeds": [50, 51, 52, 53, 54],
                    "validation_pool": "train",
                    "validation_top_k": 2
                },
                "rollout": {"budget": 75, "max_concurrent": 15, "minibatch_size": 3},
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 4,
                    "num_generations": 3,
                    "children_per_generation": 2,
                    "crossover_rate": 0.5,
                    "selection_pressure": 1.0,
                    "patience_generations": 5
                },
                "archive": {
                    "size": 64,
                    "pareto_set_size": 10,
                    "pareto_eps": 1e-6,
                    "feedback_fraction": 0.231
                },
                "token": {
                    "max_limit": 4096,
                    "counting_model": "gpt-4",
                    "enforce_limit": false
                }
            },
            "termination_config": {"max_cost_usd": 5.0, "max_trials": 75}
        }
    });

    let url = format!("{}/prompt-learning/online/jobs", backend_url.trim_end_matches('/'));

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", synth_api_key))
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "algorithm": "gepa",
            "config_body": config_body,
            "task_app_url": task_app_url,
            "task_app_api_key": task_app_api_key,
        }))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Job submission failed: {} - {}", status, body));
    }

    let result: JobResponse = response.json().await?;
    let job_id = result.job_id.or(result.id).ok_or_else(|| anyhow!("No job_id in response"))?;

    println!("✅ Job submitted: {}", job_id);
    Ok(job_id)
}

async fn poll_job(
    client: &reqwest::Client,
    backend_url: &str,
    synth_api_key: &str,
    job_id: &str,
) -> Result<bool> {
    println!("\n⏳ Polling for completion (this may take 2-5 minutes)...");

    let url = format!("{}/prompt-learning/online/jobs/{}", backend_url.trim_end_matches('/'), job_id);

    for i in 1..=120 {
        sleep(Duration::from_secs(5)).await;

        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", synth_api_key))
            .send()
            .await;

        let result: JobResponse = match response {
            Ok(resp) if resp.status().is_success() => {
                match resp.json().await {
                    Ok(r) => r,
                    Err(_) => continue,
                }
            }
            _ => continue,
        };

        let status = result.status.as_deref().unwrap_or("unknown");
        print!("\r   [{:3}/120] Status: {:<12}", i, status);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        match status {
            "succeeded" | "completed" | "success" => {
                println!();
                println!("\n✅ Job completed successfully!");

                if let Some(metadata) = result.metadata {
                    if let Some(score) = metadata.get("prompt_best_score").and_then(|v| v.as_f64()) {
                        println!("   Best Score: {:.1}%", score * 100.0);
                    }
                }
                return Ok(true);
            }
            "failed" | "cancelled" | "error" => {
                println!();
                println!("\n❌ Job failed!");
                if let Some(error) = result.error {
                    println!("   Error: {}", error);
                }
                return Ok(false);
            }
            _ => {}
        }
    }

    println!("\n⏰ Timeout - job still running after 10 minutes");
    Ok(false)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async fn run_server(state: Arc<AppState>) -> Result<()> {
    let port = state.config.port;

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/task_info", get(task_info_handler))
        .route("/rollout", post(rollout_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn run_smoke_test(config: Config, dataset: Dataset) -> Result<()> {
    println!(r#"
╭───────────────────────────────────────────────────────────────────────────╮
│  🦀 Rust GEPA Smoke Test                                                  │
╰───────────────────────────────────────────────────────────────────────────╯"#);

    let synth_api_key = config.synth_api_key.clone()
        .ok_or_else(|| anyhow!("SYNTH_API_KEY is required for smoke test"))?;
    let task_app_api_key = config.api_key.clone()
        .ok_or_else(|| anyhow!("ENVIRONMENT_API_KEY is required for smoke test"))?;

    println!("  Backend URL:     {}", config.backend_url);
    println!("  Task App Port:   {}", config.port);
    println!("");

    let state = Arc::new(AppState {
        config,
        dataset,
        http_client: reqwest::Client::new(),
    });

    // Start server in background
    let server_state = state.clone();
    let port = server_state.config.port;
    tokio::spawn(async move {
        if let Err(e) = run_server(server_state).await {
            eprintln!("Server error: {}", e);
        }
    });

    // Wait for server to start
    sleep(Duration::from_secs(1)).await;

    // Verify server is running
    let health_url = format!("http://127.0.0.1:{}/health", port);
    let client = reqwest::Client::new();

    for _ in 0..10 {
        if client.get(&health_url).send().await.is_ok() {
            println!("✓ Task app server started on port {}", port);
            break;
        }
        sleep(Duration::from_millis(500)).await;
    }

    // Create tunnel
    let (tunnel_url, mut tunnel_child) = create_tunnel(port).await?;

    // Verify tunnel is accessible
    println!("🔍 Verifying tunnel connectivity...");
    for i in 0..10 {
        match client.get(format!("{}/health", tunnel_url)).send().await {
            Ok(resp) if resp.status().is_success() => {
                println!("✓ Tunnel verified accessible");
                break;
            }
            _ => {
                if i == 9 {
                    tunnel_child.kill().await.ok();
                    return Err(anyhow!("Tunnel not accessible after 10 retries"));
                }
                sleep(Duration::from_secs(2)).await;
            }
        }
    }

    // Submit job
    let job_id = submit_job(
        &client,
        &state.config.backend_url,
        &synth_api_key,
        &tunnel_url,
        &task_app_api_key,
    )
    .await?;

    // Poll for completion
    let success = poll_job(&client, &state.config.backend_url, &synth_api_key, &job_id).await?;

    // Cleanup
    println!("\n🧹 Cleaning up...");
    tunnel_child.kill().await.ok();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if success {
        println!("✅ Rust GEPA smoke test PASSED!");
        Ok(())
    } else {
        Err(anyhow!("Rust GEPA smoke test FAILED!"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let smoke_test = args.iter().any(|a| a == "--smoke-test" || a == "-s");

    let config = Config::from_env();
    let dataset = Dataset::load();

    if smoke_test {
        run_smoke_test(config, dataset).await
    } else {
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

        run_server(state).await
    }
}
