# Changelog

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
