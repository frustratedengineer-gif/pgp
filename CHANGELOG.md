# Changelog

## [0.5.0] - Week 5 - 2026-08-12

### Added
- 6 off-the-shelf feature extractors (`src/memorylife/features/`):
  temporal (regex, deterministic), novelty (causal embedding-distance vs.
  earlier memories in the same conversation), entities (NER), intent
  (zero-shot classification), emotion, contradiction (NLI vs. the nearest
  earlier memory) -- cached per split via `features/pipeline.py`.
- Feature fusion (`src/memorylife/fusion/`): `concat` and `gated` variants;
  `cross_attention` remains a stub.
- Two new real, supervised heads: Action (store/update/merge/forget, labels
  derived from `lifecycle_event`) and Future-utility (P(retrieved again),
  labels from `observed_usage`/`no_usage_observed`) -- see
  `src/memorylife/heads/`.
- Importance "head" (`src/memorylife/heads/importance.py`): a documented
  HEURISTIC, not learned -- no ground-truth importance label exists in the
  dataset. Flagged explicitly so it's never mistaken for a trained head.
- Joint multi-task model (`src/memorylife/models/joint_predictor.py`,
  `multitask.py`, `scripts/train_joint.py`): Lifetime + Action +
  Future-utility heads trained jointly via a custom loop (not pycox's
  `CoxPH.fit()` wrapper, which can't share gradients across heads).
  **Concat fusion reaches 0.7553 +/- 0.0045 test C-index** (3 seeds), beating
  both gated fusion (0.7304 +/- 0.0082) and the Week-3/4 lone survival head
  (0.7312 +/- 0.0131) -- see `results/tables/week5_joint_model_results.md`.
- Full memory system (`src/memorylife/memory/`): `MemoryObject`, an
  in-memory brute-force vector store, forgetting policy (TTL expiry +
  Action-head "forget"), compaction (near-duplicate merging), reflection
  (importance/utility decay past predicted TTL), append-only audit log.
- Retriever (`src/memorylife/retrieval/`): similarity + importance +
  utility reranking.
- End-to-end grounded-QA inference pipeline
  (`src/memorylife/inference/`, `scripts/run_inference_demo.py`): real LLM
  call (GPT-4o via OpenRouter) over retrieved memories, demoed on a real
  conversation with genuinely conflicting facts -- the demo output is
  honest about where retrieval/prompting currently can't fully disambiguate
  (see `docs/reproducibility.md`'s known-gaps section).

### Known gaps (see `docs/reproducibility.md`)
- Fusion cross-attention variant, FAISS/Chroma store backends, SQLite
  metadata store: stubs.
- No feature-ablation or retrieval-scoring-ablation configs run yet.
- No timestamp-aware disambiguation for directly conflicting memories in
  grounded QA.
- Downstream memory-policy baselines (FIFO/LRU/mem0/MemGPT wrappers) and
  the QA-accuracy-vs-memory-size harness: not started (separate from the
  Week-5 joint model/memory system that IS built -- see `baselines/README.md`).

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
