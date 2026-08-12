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

## Known gaps / TODOs (so this doesn't read as more finished than it is)

- **Hydra wiring**: `configs/*.yaml` document the exact values Week-3/4 ran
  with, but `scripts/*.py` currently take them as argparse defaults, not
  via live Hydra/OmegaConf composition. Low risk (values match), but the
  configs aren't yet the actual source of truth at runtime.
- **`scripts/build_benchmark.py`**: empty stub. The LoCoMo/LongMemEval/
  synthetic dataset-generation pipeline is not in this repo (see
  `data/README.md`).
- **Feature extractors, fusion, other 3 heads, memory store, retrieval,
  inference** (`src/memorylife/{features,fusion,memory,retrieval,
  inference}/`): not built. Week 4 ablated encoder choice and survival-head
  hyperparameters on top of the Week-3 architecture; feature fusion and the
  other 3 heads (importance/utility/action) are Week 5's "Full System"
  scope, not built yet. Worth restating so the repo tree isn't mistaken for
  a finished joint model.
- **Brier score / IBS for non-`our_model` methods** (`results/tables/week4_richer_metrics.md`)
  use a degenerate step-function survival curve built from each baseline's
  scalar predicted-days output (see `src/memorylife/evaluation/richer_metrics.py`
  docstring) -- a legitimate way to score a point forecast under Brier's
  proper scoring rule, but it does NOT mean those baselines produce a
  calibrated probability curve the way `our_model`'s fitted Cox baseline
  hazards do. Don't quote their IBS as "model calibration" in the paper
  without this caveat.
