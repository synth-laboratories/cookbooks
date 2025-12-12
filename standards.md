# Cookbooks Style Guide

This document defines the standards for Synth Cookbooks, following OpenAI Cookbook principles. All contributors, engineers, and AI assistants should follow these guidelines when creating or updating cookbook examples.

## Philosophy

**Cookbooks are runnable examples, not tutorials.** They demonstrate real, working code that users can copy, run, and adapt. The docs provide the narrative; cookbooks provide the code.

## Directory Structure

### Official vs Dev

- **`code/`** - Official cookbooks that appear in docs
  - Must be polished, tested, and production-ready
  - Examples: Polyglot Task Apps, LangProBe benchmarks
  
- **`dev/`** - Work-in-progress cookbooks
  - Experimental or incomplete examples
  - Will eventually move to `code/` when ready

### Naming Conventions

- Use kebab-case for directory names: `banking77-polyglot`, `math-rl-qwen`
- Be descriptive but concise: `gsm8k` not `grade-school-math-8k`
- Group related examples: `langprobe/gsm8k`, `langprobe/iris`

## README.md Standards

Every cookbook **must** have a README.md that follows this structure:

### Required Sections

```markdown
# Example Name

One-line description of what this example demonstrates.

## Prerequisites

- Python 3.11+ (or language-specific version)
- `synth-ai` package installed: `pip install synth-ai`
- Required API keys (list them)
- Any other setup requirements

## Quick Start

Single command to run the example:

```bash
python main.py
```

Or for multi-language examples:

```bash
# Rust
cargo run --release

# Go
go build && ./synth-task-app

# TypeScript
npm install && npm run dev
```

## Learn More

Full walkthrough in the docs: [https://docs.usesynth.ai/cookbooks/task-apps/example-name](https://docs.usesynth.ai/cookbooks/task-apps/example-name)
```

### README Guidelines

1. **Keep it short** - Deep explanations belong in Mintlify docs, not READMEs
2. **One clear run command** - Users should be able to run it immediately
3. **Always link to docs** - README is for GitHub browsing; docs are for learning
4. **No long narratives** - Focus on "what" and "how to run", not "why" (that's in docs)

## Code Standards

### File Organization

```
example-name/
├── README.md              # Required
├── main.py                # Or task_app.py, train.py, etc.
├── config.toml           # If applicable
├── requirements.txt       # If needed (or pyproject.toml)
└── data/                  # If dataset is included
    └── dataset.json
```

### Code Quality

1. **Use `synth-ai` package** - Never use relative imports like `from ..synth_ai`
2. **Remove repo-specific code** - No hardcoded paths, no internal utilities
3. **Parameterize everything** - Use environment variables, not hardcoded values
4. **Add comments** - Explain non-obvious parts, but keep code self-documenting
5. **Error handling** - Show proper error handling patterns

### Example: Good vs Bad

**❌ Bad:**
```python
# Hardcoded path
dataset = load_dataset("/Users/dev/synth-ai/data/banking77.json")

# Relative import
from ..synth_ai.task.server import TaskAppConfig

# No error handling
result = api_call()
```

**✅ Good:**
```python
# Environment variable or parameter
dataset_path = os.getenv("DATASET_PATH", "data/banking77.json")
dataset = load_dataset(dataset_path)

# Package import
from synth_ai.task.server import TaskAppConfig

# Error handling
try:
    result = api_call()
except APIError as e:
    logger.error(f"API call failed: {e}")
    raise
```

## Mintlify Docs Integration

### MDX Page Structure

Each cookbook in `code/` must have a corresponding MDX page in `docs/cookbooks/`:

```mdx
---
title: "Example Name"
description: "Short description for search results"
---

> **Full Example:** [View on GitHub](https://github.com/synth-laboratories/cookbooks/tree/main/code/task-apps/example-name){.button}

## What you'll build

1-2 sentences describing what this example demonstrates and why it's useful.

## Prerequisites

- Python 3.11+
- `synth-ai` installed
- API keys configured

## Quick start

```bash
cd cookbooks/code/task-apps/example-name
python main.py
```

## Key code

Show only the critical parts (not the full file):

```python
from synth_ai.task.server import TaskAppConfig

def build_config() -> TaskAppConfig:
    # Core logic here
    return config
```

[View full example on GitHub](https://github.com/synth-laboratories/cookbooks/tree/main/code/task-apps/example-name)

## Related docs

- [Task Apps Guide](/task-app)
- [Prompt Optimization](/prompt-optimization/overview)
```

### Docs Guidelines

1. **Narrative wrapper** - Docs provide context; code lives in cookbooks repo
2. **Show snippets, not full files** - Highlight key parts, link to GitHub for full code
3. **Always include GitHub button** - Make it easy to find the full example
4. **Cross-link** - Link to related concepts and other cookbooks

## GitHub Links

### Link Format

Always use the cookbooks repo, not synth-ai:

```
✅ https://github.com/synth-laboratories/cookbooks/tree/main/code/task-apps/example-name
❌ https://github.com/synth-laboratories/synth-ai/tree/main/examples/...
```

### Link Targets

- **Directory links** - Point to the example directory
- **File links** - Point to specific files when showing snippets
- **Line numbers** - Use `#L123-L145` for specific code sections

## Testing Requirements

### Before Moving to `code/`

Every cookbook must:

1. **Run end-to-end** - Complete workflow works without errors
2. **Be tested** - At minimum, manual testing; automated tests preferred
3. **Have clear output** - Users can verify it's working
4. **Document dependencies** - All requirements listed

### Testing Checklist

- [ ] Example runs without errors
- [ ] All dependencies are listed
- [ ] Environment variables documented
- [ ] Output is clear and verifiable
- [ ] Works on macOS/Linux (Windows if applicable)
- [ ] No hardcoded paths or secrets

## Versioning

### Pin `synth-ai` Version

In `requirements.txt` or `pyproject.toml`:

```txt
synth-ai==0.2.25
```

**Rationale:** Examples shouldn't break when SDK updates. Pin versions and update explicitly.

### Update Strategy

When SDK releases a new version:

1. Test all cookbooks in `code/`
2. Update pinned versions
3. Fix any breaking changes
4. Update docs if API changed

## Multi-Language Examples

### Structure

For examples with multiple language implementations:

```
polyglot/
├── README.md              # Overview + language comparison
├── data/
│   └── dataset.json       # Shared dataset
├── rust/
│   ├── Cargo.toml
│   ├── src/main.rs
│   └── README.md          # Rust-specific notes
├── go/
│   ├── go.mod
│   ├── main.go
│   └── README.md          # Go-specific notes
└── typescript/
    ├── package.json
    ├── src/index.ts
    └── README.md          # TypeScript-specific notes
```

### Language-Specific READMEs

Each language subdirectory can have its own README with:
- Language-specific setup
- Build commands
- Framework notes
- Performance characteristics

## Common Patterns

### Task App Example

```python
"""Banking77 intent classification task app."""

import os
from synth_ai.task.server import TaskAppConfig, run_task_app

def build_config() -> TaskAppConfig:
    # Implementation here
    return config

if __name__ == "__main__":
    run_task_app(build_config, host="0.0.0.0", port=8001)
```

### Prompt Optimization Example

```python
"""Banking77 prompt optimization with MIPRO."""

import os
from synth_ai.prompt_learning import MIPROConfig, run_optimization

config = MIPROConfig(
    task_app_url=os.getenv("TASK_APP_URL"),
    # ... other config
)

if __name__ == "__main__":
    run_optimization(config)
```

### Configuration Files

Use TOML for configs:

```toml
[prompt_learning]
task_app_url = "http://localhost:8001"
algorithm = "mipro"

[prompt_learning.policy]
model = "gpt-4o-mini"
temperature = 0.0
```

## OpenAI Cookbook Principles

We follow OpenAI Cookbook patterns:

1. **One source of truth** - Code lives in cookbooks repo
2. **Docs are thin** - Narrative + snippets + GitHub link
3. **Every example has README** - Short, code-centric, links to docs
4. **Cross-linking** - Concepts → Cookbooks, Cookbooks → Concepts
5. **Versioning** - Pin SDK versions for stability

## Checklist for New Cookbooks

### Before Creating

- [ ] Example demonstrates a clear use case
- [ ] Code is production-quality
- [ ] No hardcoded secrets or paths
- [ ] Dependencies are minimal and documented

### During Creation

- [ ] Create directory in `code/` or `dev/`
- [ ] Write README.md following template
- [ ] Ensure code uses `synth-ai` package (not relative imports)
- [ ] Add requirements.txt or equivalent
- [ ] Test end-to-end

### Before Moving to `code/`

- [ ] All tests pass
- [ ] README follows standards
- [ ] Code is clean and well-commented
- [ ] GitHub links point to cookbooks repo
- [ ] Mintlify docs page created
- [ ] Navigation updated in `docs/docs.json`

## Examples to Reference

**Good Examples:**
- OpenAI Cookbook: https://cookbook.openai.com
- Their GitHub: https://github.com/openai/openai-cookbook

**Our Examples (when migrated):**
- `code/task-apps/polyglot/` - Multi-language example
- `code/task-apps/langprobe/gsm8k/` - Single task app example

## Questions?

- Check existing cookbooks in `code/` for patterns
- Review OpenAI Cookbook for inspiration
- Ask in #cookbooks Slack channel
- Open an issue in the cookbooks repo

## Updates

This document should be updated when:
- New patterns emerge
- Standards change
- Common mistakes are identified
- OpenAI Cookbook updates their approach

Last updated: 2025-01-XX



