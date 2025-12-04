#!/usr/bin/env bun
/**
 * GEPA Smoke Test for TypeScript Task App
 * 
 * Pure Bun/TypeScript implementation - no Python dependencies.
 * 
 * Usage:
 *   cd polyglot/typescript
 *   bun run run_smoke_test.ts
 * 
 * Environment variables (loaded from synth-ai/.env automatically):
 *   SYNTH_API_KEY       - Synth auth token
 *   ENVIRONMENT_API_KEY - Task app API key
 *   BACKEND_URL         - Backend URL (default: https://agent-learning.onrender.com/api)
 *   TASK_APP_PORT       - Local task app port (default: 8116)
 */

import { spawn } from "bun";
import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import TOML from "@iarna/toml";

// Load .env from synth-ai
const scriptDir = dirname(new URL(import.meta.url).pathname);
const synthAiEnv = resolve(scriptDir, "../../../../synth-ai/.env");

if (existsSync(synthAiEnv)) {
  const envContent = readFileSync(synthAiEnv, "utf-8");
  for (const line of envContent.split("\n")) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("#")) {
      const [key, ...valueParts] = trimmed.split("=");
      const value = valueParts.join("=").replace(/^["']|["']$/g, "");
      if (key && !process.env[key]) {
        process.env[key] = value;
      }
    }
  }
  console.log("✓ Loaded .env from synth-ai");
}

// Configuration
const BACKEND_URL = process.env.BACKEND_URL || "https://agent-learning.onrender.com/api";
const SYNTH_API_KEY = process.env.SYNTH_API_KEY;
const TASK_APP_API_KEY = process.env.TASK_APP_API_KEY || process.env.ENVIRONMENT_API_KEY;
const TASK_APP_PORT = parseInt(process.env.TASK_APP_PORT || "8116");

if (!SYNTH_API_KEY) {
  console.error("❌ SYNTH_API_KEY is required");
  process.exit(1);
}

if (!TASK_APP_API_KEY) {
  console.error("❌ TASK_APP_API_KEY or ENVIRONMENT_API_KEY is required");
  process.exit(1);
}

// Load and parse TOML config
const configPath = resolve(scriptDir, "gepa_smoke_test.toml");
if (!existsSync(configPath)) {
  console.error(`❌ Config not found: ${configPath}`);
  process.exit(1);
}

function parseToml(content: string): Record<string, any> {
  return TOML.parse(content) as Record<string, any>;
}

const configContent = readFileSync(configPath, "utf-8")
  .replace(/\$\{TASK_APP_URL\}/g, `http://localhost:${TASK_APP_PORT}`)
  .replace(/\$\{TASK_APP_API_KEY\}/g, TASK_APP_API_KEY);
const config = parseToml(configContent);

console.log("🚀 GEPA Smoke Test for TypeScript");
console.log("━".repeat(80));
console.log(`  Backend URL:     ${BACKEND_URL}`);
console.log(`  Task App Port:   ${TASK_APP_PORT}`);
console.log(`  Config:          ${configPath}`);
console.log(`  Budget:          5 rollouts (minimal smoke test)`);
console.log("");

// Start Cloudflare tunnel if backend is remote
let tunnelUrl = `http://localhost:${TASK_APP_PORT}`;
let tunnelProcess: ReturnType<typeof spawn> | null = null;

async function createQuickTunnel(port: number): Promise<string> {
  console.log("🌐 Creating Cloudflare quick tunnel...");
  
  const logFile = `/tmp/cloudflared_ts_${port}.log`;
  
  // Use Bun.spawn with proper output handling
  const proc = Bun.spawn(["cloudflared", "tunnel", "--config", "/dev/null", "--url", `http://127.0.0.1:${port}`], {
    stdout: "pipe",
    stderr: "pipe",
  });
  
  tunnelProcess = proc;
  
  // Read stderr in background (cloudflared outputs URL to stderr)
  let output = "";
  const stderrReader = proc.stderr.getReader();
  const decoder = new TextDecoder();
  
  (async () => {
    while (true) {
      const { done, value } = await stderrReader.read();
      if (done) break;
      output += decoder.decode(value, { stream: true });
    }
  })();
  
  // Wait for URL to appear in output
  for (let i = 0; i < 60; i++) {
    await Bun.sleep(500);
    const match = output.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/);
    if (match) {
      console.log(`✅ Tunnel created: ${match[0]}`);
      console.log(`   PID: ${proc.pid}`);
      return match[0];
    }
  }
  
  proc.kill();
  throw new Error(`Failed to create tunnel. Output: ${output.slice(0, 500)}`);
}

async function submitJob(taskAppUrl: string): Promise<string> {
  console.log("📤 Submitting GEPA job...");
  console.log(`   Task App URL: ${taskAppUrl}`);
  
  // Read TOML and substitute variables, then parse
  const tomlContent = readFileSync(configPath, "utf-8")
    .replace(/\$\{TASK_APP_URL\}/g, taskAppUrl)
    .replace(/\$\{TASK_APP_API_KEY\}/g, TASK_APP_API_KEY!);
  
  const parsedConfig = parseToml(tomlContent);
  
  // Build payload with config_body as parsed dict
  const payload = {
    algorithm: "gepa",
    config_body: parsedConfig,
    task_app_url: taskAppUrl,
    task_app_api_key: TASK_APP_API_KEY,
  };
  
  const submitUrl = `${BACKEND_URL.replace(/\/api\/?$/, "")}/api/prompt-learning/online/jobs`;
  
  const response = await fetch(submitUrl, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${SYNTH_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Job submission failed: ${response.status} - ${text.slice(0, 500)}`);
  }
  
  const result = await response.json();
  const jobId = result.job_id || result.id;
  
  if (!jobId) {
    throw new Error(`No job_id in response: ${JSON.stringify(result)}`);
  }
  
  console.log(`✅ Job submitted: ${jobId}`);
  return jobId;
}

async function streamEvents(jobId: string): Promise<boolean> {
  console.log("");
  console.log("⏳ Streaming events (this may take 1-2 minutes)...");
  
  const eventsUrl = `${BACKEND_URL.replace(/\/api\/?$/, "")}/api/prompt-learning/online/jobs/${jobId}/events/stream?since_seq=0`;
  const jobUrl = `${BACKEND_URL.replace(/\/api\/?$/, "")}/api/prompt-learning/online/jobs/${jobId}`;
  
  let completed = false;
  let failed = false;
  let eventCount = 0;
  let bestScore = 0;
  let errorMessage = "";
  
  // Helper to check job status
  async function checkJobStatus(): Promise<"running" | "completed" | "failed"> {
    try {
      const resp = await fetch(jobUrl, {
        headers: { "X-API-Key": SYNTH_API_KEY! },
      });
      if (resp.ok) {
        const data = await resp.json();
        const status = (data.status || "").toLowerCase();
        console.log(`   [Poll] Job status: ${status}`);
        if (status === "succeeded" || status === "completed" || status === "success") return "completed";
        if (status === "failed" || status === "cancelled" || status === "error") {
          errorMessage = data.error || data.message || "Job failed";
          return "failed";
        }
      }
    } catch (e) {
      console.log(`   [Poll] Error checking status: ${e}`);
    }
    return "running";
  }
  
  try {
    console.log("🔌 Connecting to SSE stream...");
    const response = await fetch(eventsUrl, {
      headers: {
        "X-API-Key": SYNTH_API_KEY!,
        "Accept": "text/event-stream",
      },
    });
    
    if (!response.ok) {
      throw new Error(`SSE failed: ${response.status}`);
    }
    
    console.log("✅ SSE stream connected");
    
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");
    
    const decoder = new TextDecoder();
    let buffer = "";
    let lastStatusCheck = Date.now();
    
    const streamTimeout = setTimeout(() => {
      console.log("   ⏰ Stream timeout reached, will poll for status");
    }, 300000); // 5 minute timeout
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log("   📡 SSE stream ended, will poll for final status...");
        clearTimeout(streamTimeout);
        break;
      }
      
      buffer += decoder.decode(value, { stream: true });
      
      while (buffer.includes("\n\n")) {
        const [message, rest] = buffer.split("\n\n", 2);
        buffer = rest || "";
        
        if (!message.trim()) continue;
        
        for (const line of message.split("\n")) {
          if (line.startsWith("data: ")) {
            try {
              const event = JSON.parse(line.slice(6));
              eventCount++;
              
              const eventType = event.type || "";
              
              // Log ALL events for debugging
              const acc = event.data?.accuracy || event.accuracy;
              const msg = event.message || event.data?.message || "";
              if (acc) {
                console.log(`   📊 Event ${eventCount}: ${eventType} (accuracy: ${(acc * 100).toFixed(1)}%)`);
              } else if (msg) {
                console.log(`   📝 Event ${eventCount}: ${eventType} - ${msg.slice(0, 100)}`);
              } else {
                console.log(`   📌 Event ${eventCount}: ${eventType}`);
              }
              
              // Track best score
              const accuracy = event.data?.accuracy || event.accuracy;
              if (typeof accuracy === "number" && accuracy > bestScore) {
                bestScore = accuracy;
              }
              
              // Check for FINAL job completion (not intermediate phase completions)
              // Only prompt.learning.completed or prompt.learning.finished indicates job done
              if (eventType === "prompt.learning.completed" || 
                  eventType === "prompt.learning.finished" ||
                  eventType === "prompt.learning.succeeded") {
                console.log(`   ✅ Job completion event: ${eventType}`);
                completed = true;
                break;
              }
              
              if (eventType.includes("failed") || eventType.includes("error")) {
                errorMessage = event.message || event.data?.message || event.data?.error || "Unknown error";
                console.log(`   ❌ Job failed: ${errorMessage}`);
                failed = true;
                break;
              }
            } catch {}
          }
        }
        
        if (completed || failed) break;
      }
      
      if (completed || failed) break;
      
      // Periodic status check every 10s
      if (Date.now() - lastStatusCheck > 10000) {
        const status = await checkJobStatus();
        if (status === "completed") { completed = true; break; }
        if (status === "failed") { failed = true; break; }
        lastStatusCheck = Date.now();
      }
    }
    
    reader.releaseLock();
    
  } catch (e) {
    console.log(`⚠️  SSE error: ${e}`);
  }
  
  // If stream ended without completion/failure, poll for final status
  if (!completed && !failed) {
    console.log("   🔄 Polling for final status (job may still be running)...");
    for (let i = 0; i < 120; i++) {  // 10 minutes max polling
      const status = await checkJobStatus();
      if (status === "completed") { 
        completed = true; 
        console.log("   ✅ Job completed successfully!");
        break; 
      }
      if (status === "failed") { 
        failed = true; 
        break; 
      }
      await Bun.sleep(5000);
      if (i % 6 === 0 && i > 0) console.log(`   ⏳ Still polling... (${i * 5}s elapsed)`);
    }
  }
  
  console.log("");
  console.log(`   Events received: ${eventCount}`);
  if (bestScore > 0) {
    console.log(`   Best score: ${(bestScore * 100).toFixed(1)}%`);
  }
  if (errorMessage) {
    console.log(`   Error: ${errorMessage}`);
  }
  
  return completed && !failed;
}

async function main() {
  try {
    // Check if task app is running
    try {
      const healthResp = await fetch(`http://localhost:${TASK_APP_PORT}/health`);
      if (!healthResp.ok) throw new Error("unhealthy");
      console.log(`✓ Task app running on port ${TASK_APP_PORT}`);
    } catch {
      console.error(`❌ Task app not running on port ${TASK_APP_PORT}`);
      console.error(`   Start it with: PORT=${TASK_APP_PORT} ENVIRONMENT_API_KEY=... bun run dev`);
      process.exit(1);
    }
    
    // Create tunnel if backend is remote
    if (BACKEND_URL.startsWith("https://")) {
      console.log("⚠️  Backend is remote, creating tunnel...");
      tunnelUrl = await createQuickTunnel(TASK_APP_PORT);
      console.log("⏳ Waiting for tunnel to stabilize (3s)...");
      await Bun.sleep(3000);
    }
    
    // Submit job
    const jobId = await submitJob(tunnelUrl);
    
    // Stream events
    const success = await streamEvents(jobId);
    
    // Cleanup
    if (tunnelProcess) {
      console.log("🧹 Cleaning up tunnel...");
      tunnelProcess.kill();
    }
    
    console.log("");
    console.log("━".repeat(80));
    if (success) {
      console.log("✅ TypeScript GEPA smoke test PASSED!");
      process.exit(0);
    } else {
      console.log("❌ TypeScript GEPA smoke test FAILED!");
      process.exit(1);
    }
    
  } catch (e) {
    console.error(`❌ Error: ${e}`);
    if (tunnelProcess) tunnelProcess.kill();
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(`❌ Unhandled error: ${e}`);
  process.exit(1);
});

