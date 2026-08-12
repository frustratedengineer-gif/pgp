# Reproducibility

## Hardware / software used for the Week-3 results

- 8x NVIDIA H200 (140GB) shared GPU node, CUDA driver 550.90, CUDA 12.4
- Python 3.10.12, PyTorch 2.6.0+cu124 (see `requirements.txt` for the full
  pinned environment)
- The node is **shared** with other users/jobs. GPU indices were chosen at
  runtime based on which device had headroom, not hardcoded -- if
  re-running on the same kind of shared node, check `nvidia-smi` first
  rather than assuming a device is free.

## Runtimes (single seed, single GPU)

| Step | Time |
|---|---|
| Embed all 10,152 memories (BGE-base) | ~10s |
| Train survival model (CoxPH, early-stopped ~epoch 20) | ~30s |
| `bucket_classifier` baseline (fit + predict all splits) | <5s |
| `heuristic_ttl` baseline (all splits) | <5s |
| `llm_prompted_ttl` baseline, val (939 records, 6 workers) | 86s |
| `llm_prompted_ttl` baseline, test (916 records) | 78s |
| `llm_prompted_ttl` baseline, train (8297 records) | 710s (~12 min) |

## Week-4 runtimes (5-seed sweep, single GPU, embeddings/data already cached)

| Step | Time |
|---|---|
| Train + evaluate our_model + bucket_classifier, one seed | ~15-25s |
| Full 5-seed sweep (`scripts/run_seed_sweep.py`) | ~2 min |
| Encoder ablation, 2 encoders x 5 seeds (`scripts/run_ablations.py`) | ~3 min (+ one-time BGE-large download/encode) |
| Hyperparameter sensitivity, 7 variants x 3 seeds | ~4 min |
| Bootstrap significance, 1000 resamples x 6 methods x 2 splits (`scripts/compute_significance.py`) | ~1 min |
| Richer metrics: time-dependent AUC + Brier/IBS (`scripts/compute_richer_metrics.py`) | ~15s |
| Consistency check: 100 records x 3 paraphrases x 3 methods, 900 calls, cache-assisted (`scripts/consistency_check.py`) | ~10-15 min (dominated by Gemini's per-call latency) |

## Seeds

Week-3 results used **seed=42 only**. Week 4 ran the full 5-seed sweep
(13, 42, 1337, 2024, 7) from `experiments/seeds.txt` and reports mean ± std
-- see `results/tables/week4_multiseed_results.md`. Our model:
0.7312 ± 0.0131 (test), 0.7237 ± 0.0063 (val) -- low variance, not a lucky
single-seed draw. `memorylife.utils.seeding.seed_everything()` covers
`random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`.

Known non-determinism not yet addressed:
- `torch.use_deterministic_algorithms(True)` is not set.
- The LLM baselines use `temperature=0.0` (approximately deterministic per
  provider, but not guaranteed bit-identical across API versions/hardware).
  Raw responses ARE now cached per (model, prompt) under
  `artifacts/llm_cache/` (see `baselines/_openrouter_client.py`), so a
  re-run of the same experiment doesn't re-query or re-bill -- this was the
  Week-3 gap flagged here, closed in Week 4.

## Week-5 runtimes (single GPU)

| Step | Time |
|---|---|
| Feature pipeline: 6 extractors over all ~10,152 records (`scripts/*` via `features/pipeline.ensure_features`) | ~163s (one-time, cached to `artifacts/features/`) |
| Joint model training, one seed/fusion variant (`scripts/train_joint.py`) | ~1-3 min (early-stopped, 70-90 epochs) |
| `scripts/run_inference_demo.py`, one 10-memory conversation, 4 grounded-QA queries | ~15-20s (dominated by 4 GPT-4o calls) |

## Known gaps / TODOs (so this doesn't read as more finished than it is)

- **Hydra wiring**: `configs/*.yaml` document the exact values Week-3/4/5
  ran with, but `scripts/*.py` currently take them as argparse defaults,
  not via live Hydra/OmegaConf composition. Low risk (values match), but
  the configs aren't yet the actual source of truth at runtime.
- **`scripts/build_benchmark.py`**: empty stub. The LoCoMo/LongMemEval/
  synthetic dataset-generation pipeline is not in this repo (see
  `data/README.md`).
- **Brier score / IBS for non-`our_model` methods** (`results/tables/week4_richer_metrics.md`)
  use a degenerate step-function survival curve built from each baseline's
  scalar predicted-days output (see `src/memorylife/evaluation/richer_metrics.py`
  docstring) -- a legitimate way to score a point forecast under Brier's
  proper scoring rule, but it does NOT mean those baselines produce a
  calibrated probability curve the way `our_model`'s fitted Cox baseline
  hazards do. Don't quote their IBS as "model calibration" in the paper
  without this caveat.
- **Importance head is a heuristic, not learned** (`src/memorylife/heads/importance.py`):
  no ground-truth importance label exists anywhere in the dataset schema.
  If a real signal becomes available (engagement logs, explicit ratings),
  replace it with a trained head using the Action/Future-utility heads'
  pattern -- don't cite the current heuristic as "the model learned what's
  important."
- **Fusion cross-attention variant** (`src/memorylife/fusion/cross_attention.py`):
  stub. Only `concat`/`gated` were built and compared; concat won (see
  `results/tables/week5_joint_model_results.md`) but cross-attention wasn't
  tried, so "concat is best" is only established against the two fusion
  mechanisms actually implemented.
- **Retrieval-scoring ablation not run**: `configs/retrieval/sim_only.yaml`
  (alpha=beta=0) is documented but `scripts/run_inference_demo.py` only
  ever exercises `sim_importance_utility.yaml`'s weights -- "does
  importance/utility reranking actually improve retrieval" is not yet
  measured, only assumed.
- **No timestamp-aware disambiguation in grounded QA**: when the retriever
  surfaces two directly conflicting memories with near-identical scores
  (e.g. two different phone numbers), the LLM prompt doesn't include
  timestamps or explicit recency signal, so it can't reliably resolve which
  is current -- observed directly in `scripts/run_inference_demo.py`'s
  default demo (see `docs/architecture.md`'s Week-5 section). The Action
  head's `update`/`merge` predictions are meant to prevent stale
  duplicates from coexisting in the first place, but this specific demo
  conversation has genuine cases the current heuristics don't fully
  resolve -- a known, undisguised limitation, not a bug being hidden.
- **`faiss_store.py`/`chroma_store.py`/`sqlite_metadata.py`**: stubs. The
  working default (`memory/store/numpy_store.py`) is brute-force cosine
  search, which is fine at MemoryLifeBench's scale (~10K memories) but
  won't scale past that without a real ANN index.
- **Feature-extractor ablation configs** (`configs/features/no_*.yaml` per
  `docs/repo_structure_reference.md`'s template): not created. Which of the
  6 extractors actually drive the joint model's C-index/action-acc gains
  vs. which are dead weight is not yet measured.
