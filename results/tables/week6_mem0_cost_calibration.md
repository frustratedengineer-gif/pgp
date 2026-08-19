# Mem0 indexing cost calibration (GPT-4o via OpenRouter)

Reviewer-gap #1 follow-up, and a fix for a real gap the final consistency
pass caught: `scripts/calibrate_mem0_cost.py` originally only printed its
result, so the $0.0106/turn and ~$62 figures quoted in `paper/draft.md`
Section 6.15 and `README.md` had no committed backing table -- this is
that table, reconstructed from the actual calibration run's real
before/after OpenRouter balance (`/api/v1/credits`), not re-estimated.

Real run: `conv-30`, capped at the first 60 turns (a calibration run,
not the full eval -- see the script's own `--max-turns` default), Mem0's
`openai` LLM provider auto-routed through OpenRouter to `openai/gpt-4o`.

| Metric | Value |
|---|---|
| Turns indexed | 60 |
| Starting balance | $1.7268 |
| Ending balance | $1.0887 |
| Spent | $0.6381 |
| $/turn | $0.01063 |
| Wall-clock | 110.8s (1.85s/turn, GPT-4o API latency, not local compute) |
| Extrapolated cost for all 5,882 LoCoMo turns (10 conversations) | **$62.55** |

This is why the real (free) Mem0 baseline run (`week6_mem0_baseline.md`)
used a local Qwen2.5-7B-Instruct server instead of GPT-4o for Mem0's own
indexing calls -- see `baselines/mem0_wrapper.py`'s module docstring for
the full substitution and its disclosed limitations.
