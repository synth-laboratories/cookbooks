# Synth Cookbooks

Example scripts and tutorials for using Synth's APIs and tools.

## Quick Start: GEPA Crafter Demo

Run GEPA prompt optimization on the Crafter game - no API key needed!

```bash
pip install httpx
python gepa_demo.py --nickname "YourName"
```

Watch live with visualization: **https://www.usesynth.ai/blog/gepa-for-agents**

---

## Ready

**Live Demos**
- [`gepa_demo.py`](./gepa_demo.py) — GEPA for Crafter (runs against prod, no auth needed)
- `code/demos/banking77/` — Banking77 intent classification with GEPA

**GEPA & MIPRO In-Process**
- `code/training/prompt_learning/gepa/` — GEPA walkthrough
- `code/training/prompt_learning/mipro/` — MIPRO walkthrough

**Polyglot Task Apps**
- `code/polyglot/typescript/` — TypeScript (Bun) — tested with GEPA, 80% accuracy
- `code/polyglot/rust/` — Rust (Axum) — tested with GEPA, 70% accuracy
- `code/polyglot/go/` — Go (stdlib)
- `code/polyglot/python/` — Python (Flask)

## Pending

- `code/training/prompt_learning/sdk/` — SDK examples
- `code/training/prompt_learning/cli/` — CLI walkthrough
- `code/training/sft/` — SFT cookbook
- `code/training/rl/` — RL cookbook

## GEPA Demo Options

```
python gepa_demo.py --help

Options:
  --nickname, -n     Your name on the leaderboard (max 32 chars)
  --generations, -g  Optimization generations 1-5 (default: 2)
  --population, -p   Candidates per generation 2-10 (default: 3)
  --model, -m        Policy model (default: openai/gpt-oss-20b)
  --list             List active demos and exit
  --quiet, -q        Suppress progress output

Available models:
  - openai/gpt-oss-20b (default, faster)
  - openai/gpt-oss-120b (larger)
  - qwen/qwen3-32b
  - moonshotai/kimi-k2-instruct
```

## Resources

- [GEPA Blog Post](https://www.usesynth.ai/blog/gepa-for-agents)
- [Synth Documentation](https://docs.usesynth.ai)
