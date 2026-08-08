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

## Seeds

Week-3 results use **seed=42 only**. `experiments/seeds.txt` lists 5 seeds
(13, 42, 1337, 2024, 7) a full paper run should average over with mean ±
std -- not done yet. `memorylife.utils.seeding.seed_everything()` covers
`random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`.

Known non-determinism not yet addressed:
- `torch.use_deterministic_algorithms(True)` is not set.
- The LLM baseline uses `temperature=0.0` (approximately deterministic for
  local model serving, but not guaranteed bit-identical across ollama
  versions/hardware). Raw responses are not currently cached per-record
  (a re-run re-queries the model); adding response caching is a good
  Week-4 addition once the calibration/consistency experiments start
  re-querying the same records repeatedly.

## Known gaps / TODOs (so this doesn't read as more finished than it is)

- **Hydra wiring**: `configs/*.yaml` document the exact values Week-3 ran
  with, but `scripts/*.py` currently take them as argparse defaults, not
  via live Hydra/OmegaConf composition. Low risk (values match), but the
  configs aren't yet the actual source of truth at runtime.
- **`scripts/build_benchmark.py`**: empty stub. The LoCoMo/LongMemEval/
  synthetic dataset-generation pipeline is not in this repo (see
  `data/README.md`).
- **`tests/`**: all files are empty stubs. No automated test coverage yet,
  including the one the template explicitly calls out as important:
  censoring must be handled correctly by the loss (`test_censoring.py`,
  `test_survival_loss.py`).
- **Feature extractors, fusion, other 3 heads, memory store, retrieval,
  inference** (`src/memorylife/{features,fusion,memory,retrieval,
  inference}/`): not built. Week 3 trained the Lifetime/survival head
  alone on raw BGE embeddings, no feature fusion. This matches the plan
  (features/fusion are Week 4's ablation surface) but is worth restating
  so the repo tree isn't mistaken for a finished joint model.
