# Walkthrough Documentation Status

## ✅ Completed (2025-11-30)

### GEPA (Banking77)

| File | Status | Notes |
|------|--------|-------|
| `gepa/walkthrough.md` | ✅ Updated | Real logs from job `pl_5ea04259c2fd4c7a` |
| `gepa/results.json` | ✅ Updated | Best score: 83.33% |
| `gepa/run_walkthrough.py` | ✅ Working | Runs end-to-end successfully |
| `gepa/gepa_output.log` | ✅ Created | Full execution logs |

**Results:** Job `pl_5ea04259c2fd4c7a` - 83.33% accuracy, ~35s, llama-3.1-8b-instant

### MIPRO (Banking77)

| File | Status | Notes |
|------|--------|-------|
| `mipro/walkthrough.md` | ✅ Updated | Real logs from job `pl_e95cc778c0fb4742` |
| `mipro/results.json` | ✅ Created | Best score: 60% |
| `mipro/run_walkthrough.py` | ✅ Created | Runs end-to-end successfully |
| `mipro/mipro_output.log` | ✅ Created | Full execution logs |

**Results:** Job `pl_e95cc778c0fb4742` - 60% accuracy, ~130s, gpt-4o-mini

---

## ⏳ Pending Walkthroughs

### ✅ GEPA with Hosted Judge (Completed 2025-11-30)

| File | Status | Notes |
|------|--------|-------|
| `gepa/run_walkthrough_judge.py` | ✅ Created | Script for judge-enabled GEPA |
| `gepa/results_judge.json` | ✅ Created | Best score: 71.43% |
| `gepa/gepa_judge_output.log` | ✅ Created | Full execution logs |

**Results:** Job `pl_e0b296b671714328` - 71.43% fused score, ~45s, llama-3.1-8b-instant + gpt-5-mini judge

**Config features used:**
- `reward_source = "fused"` - Combines task_app accuracy + judge quality
- `weight_env = 1.0`, `weight_outcome = 0.25` - Fusion weights
- `backend_model = "gpt-5-mini"` - OpenAI judge model

### ⏳ In Progress: GEPA for Crafter

Uses `dev/blog_posts/langprobe/task_specific/crafter/crafter_gepa.toml` for agent prompt optimization in a game environment.

**Status:** Working on output extraction bug - extraction logic needs to handle Crafter's `"actions"` tool call format.

**Files created:**
- ✅ `crafter/` directory under `code/training/prompt_learning/`
- ✅ `crafter/run_walkthrough.py`
- ✅ `crafter/crafter_task_app.py`
- ⏳ `crafter/walkthrough.md` - Pending successful run
- ⏳ `crafter/config.toml` - Needs finalization

**Known Issue:** Output extraction bug documented in `crafter/gepa_output_bug.txt` - extraction logic is too restrictive and doesn't handle non-standard tool call argument names.

### ⏳ Pending: SDK Examples

| File | Status |
|------|--------|
| `sdk/basic.py` | ⏳ Pending |
| `sdk/advanced.py` | ⏳ Pending |
| `sdk/in_process.py` | ⏳ Pending |

### ✅ Polyglot Task Apps (Completed)

Different language implementations in `code/polyglot/` - all have complete walkthroughs and successful runs:

| Language | Walkthrough | Results | Status |
|----------|-------------|---------|--------|
| Python | ✅ `python/walkthrough.md` | Job `pl_7e0227cc41454ec5` - 66.7% accuracy | ✅ Complete |
| Rust | ✅ `rust/walkthrough.md` | Job `pl_4f69a1b099a14e4b` - 100% accuracy | ✅ Complete |
| TypeScript | ✅ `typescript/walkthrough.md` | Job `pl_787c47998cfe4745` - 85.7% accuracy | ✅ Complete |
| Go | ✅ `go/walkthrough.md` | Job `pl_1dd94dfdc8c6479d` - 60% accuracy | ✅ Complete |

All polyglot examples demonstrate GEPA optimization with task apps written in different languages, showing that the SDK works with any HTTP-compatible task app regardless of implementation language.

### ⏳ Pending: CLI Walkthrough

- `cli/walkthrough.md` — Needs real execution logs
- `cli/run_gepa.sh` — Document expected output
- `cli/run_mipro.sh` — Document expected output

---

## Bug Fixes Applied

### Judge API URL path missing /api prefix (2025-11-30)
- **Commit**: `7dc5aba8` pushed to `dev`
- **Files**:
  - `backend/app/routes/prompt_learning/routes_online.py`
  - `backend/app/routes/prompt_learning/core/rubric_pipeline.py`
- **Fix**: Changed `/judge/v1/score` → `/api/judge/v1/score`
- **Root cause**: Judge routes mounted at `/api/judge/v1` but validation code constructed URL without `/api` prefix

### TokenRates subscript error (2025-11-30)
- **Commit**: `788ee021` pushed to `dev`
- **File**: `backend/app/routes/prompt_learning/algorithm/mipro/optimizer/optimizer.py`
- **Fix**: Changed `model_data["input"]` → `model_data.input_usd`

---

## Other Environments (in `dev/`)

These task apps exist but are in the dev directory, not the main cookbooks:

| Environment | Location | Notes |
|-------------|----------|-------|
| Crafter | `dev/task_apps/crafter/` | Game environment for RL |
| HotpotQA | `dev/blog_posts/gepa/` | Multi-hop QA |
| IFBench | `dev/blog_posts/gepa/` | Instruction following |
| HoVer | `dev/blog_posts/gepa/` | Claim verification |
| Enron | `dev/task_apps/enron/` | Email classification |
| Verilog | `dev/task_apps/verilog/` | Hardware description |
| HeartDisease | `dev/blog_posts/dspy_scaling_laws/` | Medical classification |

---

## How to Run Completed Walkthroughs

### GEPA (Banking77)
```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run python /Users/joshpurtell/Documents/GitHub/cookbooks/code/training/prompt_learning/gepa/run_walkthrough.py
```

### MIPRO (Banking77)
```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run python /Users/joshpurtell/Documents/GitHub/cookbooks/code/training/prompt_learning/mipro/run_walkthrough.py
```

---

## Next Steps

1. ✅ **GEPA with Judge walkthrough** - Completed
2. ⏳ **Fix Crafter GEPA** - Working on output extraction bug
3. **Document SDK examples** - Add real execution examples
4. ✅ **Polyglot walkthroughs** - All complete (Python, Rust, TypeScript, Go)

---

Last updated: 2025-11-30
