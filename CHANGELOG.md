# Changelog

## [0.4.0] - Week 4 - 2026-08-12

### Added
- Two real-frontier-LLM baselines, `chatgpt_prompted_ttl` (GPT-4o) and
  `gemini_prompted_ttl` (Gemini 2.5 Pro), both via OpenRouter
  (`baselines/_openrouter_client.py`) with per-call token-usage tracking
  (`results/tables/llm_token_usage.md`) and on-disk response caching.
- 5-seed reproduction of the Week-3 result (`scripts/run_seed_sweep.py`,
  `results/tables/week4_multiseed_results.md`): our model
  0.7312 ± 0.0131 (test) / 0.7237 ± 0.0063 (val).
- Bootstrap significance testing (`src/memorylife/evaluation/significance.py`,
  `scripts/compute_significance.py`): our model beats every baseline at
  p < 0.001 on both splits (`results/tables/week4_significance.md`).
- Metrics beyond C-index: time-dependent AUC, Brier score, Integrated Brier
  Score (`src/memorylife/evaluation/richer_metrics.py`,
  `scripts/compute_richer_metrics.py`, new `scikit-survival` dependency).
- LLM prediction consistency check across paraphrased prompts
  (`scripts/consistency_check.py`, `results/tables/week4_consistency.md`):
  GPT-4o/Gemini are noticeably more stable under rewording (CV ~0.38) than
  the local Qwen2.5-7B baseline (CV ~0.75).
- Lightweight ablations (`scripts/run_ablations.py`): encoder choice
  (BGE-base vs BGE-large) and survival-head hyperparameter sensitivity
  (`results/tables/week4_ablation_encoder.md`, `week4_ablation_hparams.md`).
- Real test coverage: `tests/test_censoring.py`, `tests/test_survival_loss.py`
  (previously empty stubs).

### Known gaps (see `docs/reproducibility.md`)
- Dataset-generation code (`scripts/build_benchmark.py`) not yet written.
- Hydra config wiring not yet live (configs document actual run values only).
- Features/fusion/other 3 heads/memory/retrieval/inference: Week 5 scope,
  not started.

## [0.3.0] - Week 3 - 2026-08-08

### Added
- Survival target derivation (`memorylife.data.event_labeling` /
  `censoring`): (T, delta) from `injected_at`/`invalidated_at`/`probes`,
  documented administrative-censoring convention for real-source records
  with no probes.
- BGE-base-en-v1.5 embedding pipeline (`memorylife.encoders`).
- Lifetime head: CoxPH survival model on frozen embeddings
  (`memorylife.heads.survival`, `memorylife.losses.cox_partial`).
- Three baselines: `llm_prompted_ttl` (local Qwen2.5-7B via ollama),
  `bucket_classifier` (day/week/permanent logistic regression),
  `heuristic_ttl` (regex/keyword rules).
- C-index evaluation and results table (`results/tables/week3_results_table.md`).
- First results: our model 0.72 test C-index vs. 0.63 / 0.52 / 0.48 for the
  three baselines.
- Repository restructured to match `docs/` reviewer-facing layout (this
  changelog entry included).

### Known gaps (see `docs/reproducibility.md`)
- Dataset-generation code (`scripts/build_benchmark.py`) not yet written.
- No automated tests yet.
- Hydra config wiring not yet live (configs document actual run values only).

## [0.2.0] - Week 2 - dataset + MemoryLifeBench v0.1
- Synthetic conversation generation with injected facts, scheduled probes,
  lifecycle events (update/contradiction).
- Merged real (LoCoMo, LongMemEval) + synthetic into train/val/test splits.

## [0.1.0] - Week 1 - dataset from real conversations
- Candidate memory extraction from LoCoMo/LongMemEval.
- Lifetime labeling and censoring flags for real-source records.
