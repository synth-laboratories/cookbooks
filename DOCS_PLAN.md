# Mintlify Docs Plan

Two new cookbook pages for `monorepo/docs/cookbooks/`.

---

## 1. In-Process GEPA/MIPRO

**File:** `cookbooks/prompt-optimization-in-process.mdx`

**Source:** `cookbooks/code/training/prompt_learning/gepa/` and `mipro/`

### Structure

```
1. Intro (2 sentences)
2. Prerequisites (env vars, uv)
3. Quick Start
   - GEPA one-liner
   - MIPRO one-liner
4. What Happens
   - Task app starts in background
   - Tunnel created automatically
   - Job runs, polls, returns results
5. Code Walkthrough
   - run_walkthrough.py snippet
   - InProcessTaskApp usage
6. Config Reference
   - Key GEPA params (population, generations, mutation)
   - Key MIPRO params (iterations, demos, instructions)
7. Results
   - Show actual job IDs and scores from results.json
8. Troubleshooting (3-4 bullets)
```

### Content to Pull

From `gepa/walkthrough.md`:
- Real job ID: `pl_5ea04259c2fd4c7a`
- Score: 83.33%
- Time: ~35s

From `mipro/walkthrough.md`:
- Real job ID: `pl_e95cc778c0fb4742`
- Score: 60%
- Time: ~130s

From `gepa/run_walkthrough.py`:
- InProcessTaskApp pattern
- PromptLearningJob.from_config usage

---

## 2. Polyglot Task Apps via CLI

**File:** `cookbooks/prompt-optimization-polyglot.mdx`

**Source:** `cookbooks/code/polyglot/`

### Structure

```
1. Intro (why polyglot - 2 sentences)
2. Supported Languages
   - Python, Rust, TypeScript, Go (table with status)
3. Quick Start per Language
   - Build command
   - Run command
   - Tunnel command
   - CLI train command
4. Results Summary
   - Table: language, job_id, accuracy, time
5. The Contract
   - Link to OpenAPI spec
   - Required endpoints (/health, /rollout)
6. Running Optimization
   - `synth train config.toml --poll`
   - No Python SDK required
7. Troubleshooting (3-4 bullets)
```

### Content to Pull

From `polyglot/*/results.json`:

| Language   | Job ID                  | Accuracy |
|------------|-------------------------|----------|
| Python     | `pl_7e0227cc41454ec5`   | 66.7%    |
| Rust       | `pl_4f69a1b099a14e4b`   | 100%     |
| TypeScript | `pl_787c47998cfe4745`   | 85.7%    |
| Go         | `pl_1dd94dfdc8c6479d`   | 60%      |

From `polyglot/*/walkthrough.md`:
- Build/run commands
- Tunnel setup
- CLI invocation

---

## Navigation Update

Update `docs.json` Cookbooks tab:

```json
{
  "group": "Prompt Optimization",
  "pages": [
    "cookbooks/prompt-optimization-in-process",
    "cookbooks/prompt-optimization-polyglot",
    "cookbooks/prompt-optimization-mipro",
    "cookbooks/prompt-optimization-gepa"
  ]
}
```

---

## Existing Docs to Reference

- `prompt-optimization/gepa-in-process.mdx` — detailed GEPA mechanics
- `prompt-optimization/polyglot-task-apps.mdx` — full polyglot reference
- `prompt-optimization/walkthroughs/in-process-task-app.mdx` — step-by-step

New cookbook pages should be **terse** and link to these for details.

---

## Style

- Max 150 lines per page
- Code blocks with real commands
- Tables for comparisons
- No fluff, direct instructions
- Link to source files in cookbooks repo

