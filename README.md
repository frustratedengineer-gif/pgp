# MemoryLifeBench

Memory lifetime prediction as a **time-to-event problem**. Instead of
asking "should I store this memory?", we ask "how long should I keep this
memory?" -- and train a survival model, not a classifier, to answer it.

Architecture figure: [`docs/figures/architecture.pdf`](docs/figures/architecture.pdf)
(GitHub renders PDFs on click; there's no PNG/SVG export yet -- see
`docs/architecture.md` for a box-by-box walkthrough of what's built vs.
still a stub).

**Status: Week 4 of 6.** Dataset (W1), benchmark (W2), the first trained
survival model + baseline comparison (W3), and multi-seed/significance/
ablation/consistency experiments (W4) are done. Weeks 5-6 (the end-to-end
memory system, the paper) are not started -- see `docs/reproducibility.md`
"Known gaps" for the full honest list before assuming more of this repo
works than actually does.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n memorylife python=3.10
pip install -r requirements.txt                          # CUDA 12.4 build of torch; edit the --extra-index-url line for a different CUDA version
pip install -e .
```

## Quickstart (< 5 minutes, no GPU required, no data download needed)

Runs the full pipeline (preprocess -> train -> baselines) on the 60 records
committed in `data/samples/`:

```bash
bash scripts/run_smoke.sh
```

## Full data

`data/raw/{train,val,test}.jsonl` are not committed (see `data/README.md`
and `LICENSE-DATA` for why). If you have them, drop them in
`data/raw/` and verify with:

```bash
bash data/download.sh   # currently just checksum-verifies data/raw/, see the TODO in the script
```

## Reproduce the Week-3 results

```bash
bash scripts/run_all.sh
```

Expected runtime: ~15 minutes on a single modern GPU (dominated by the LLM
baseline over the full train split). Full breakdown, hardware, and known
non-determinism in `docs/reproducibility.md`.

Week-4 experiments are separate scripts, not part of `run_all.sh` (they're
significantly more expensive -- e.g. the consistency check alone makes ~900
LLM calls): `scripts/run_seed_sweep.py`, `scripts/compute_significance.py`,
`scripts/compute_richer_metrics.py`, `scripts/run_ablations.py`,
`OPENROUTER_API_KEY=... scripts/consistency_check.py`. Runtimes for each in
`docs/reproducibility.md`.

## Results (Week 3: our survival model vs. 5 baselines, by C-index)

| Split | Method | C-index |
|---|---|---|
| test | **Our survival model (CoxPH on BGE embeddings)** | **0.7218** |
| test | Day/week/permanent classifier | 0.6298 |
| test | LLM-prompted TTL (ChatGPT / GPT-4o) | 0.5411 |
| test | LLM-prompted TTL (Qwen2.5-7B, local) | 0.5207 |
| test | LLM-prompted TTL (Gemini) | 0.4806 |
| test | Recency-frequency heuristic | 0.4753 |
| val | **Our survival model (CoxPH on BGE embeddings)** | **0.7134** |
| val | Day/week/permanent classifier | 0.6195 |
| val | LLM-prompted TTL (Qwen2.5-7B, local) | 0.5713 |
| val | LLM-prompted TTL (ChatGPT / GPT-4o) | 0.5312 |
| val | Recency-frequency heuristic | 0.4849 |
| val | LLM-prompted TTL (Gemini) | 0.4284 |

Full table: `results/tables/week3_results_table.md`. C-index = 0.5 is
random; 1.0 is perfect ranking. The core result: our trained survival model
beats not just a local 7B model but real GPT-4o and Gemini prompted for the
same task -- zero-shot lifetime prediction from an LLM, even a frontier
one, is close to a coin flip on this benchmark; a model trained
specifically on the survival objective is not.

**Token spend for the GPT-4o/Gemini baselines** (cost/efficiency, tracked as
a secondary metric alongside C-index): `results/tables/llm_token_usage.md`.
GPT-4o used ~139 tokens/call and finished val+test (1,855 zero-shot calls)
in ~10 minutes; Gemini 2.5 Pro used ~113 tokens/call but ~5x the wall time
per call (reasoning overhead), and still landed a lower C-index than the
free local model on both splits.

## Week-4 experiments: is the Week-3 result actually solid?

The single-seed Week-3 numbers above could in principle have been a lucky
draw. Week 4 checks that, and looks for cracks beyond raw C-index:

| Question | Answer | Table |
|---|---|---|
| Does it hold across seeds? | Yes: 0.7312 +/- 0.0131 (test), 0.7237 +/- 0.0063 (val) over 5 seeds -- low variance | `week4_multiseed_results.md` |
| Is "ours beats every baseline" statistically significant? | Yes, p < 0.001 for every baseline on both splits (bootstrap, 1000 resamples) | `week4_significance.md` |
| Does it hold on metrics beyond C-index? | Yes: also best time-dependent AUC (0.79 test) and lowest (best) Integrated Brier Score (0.25 test) | `week4_richer_metrics.md` |
| Is a bigger/different encoder or head size the real driver? | Marginal: BGE-large edges out BGE-base slightly (0.738 vs 0.731 test), hyperparameter variants stay within ~0.72-0.75 -- the Week-3 defaults weren't a lucky pick | `week4_ablation_encoder.md`, `week4_ablation_hparams.md` |
| Are the LLM baselines at least *consistent*, even if less accurate? | Mixed: GPT-4o/Gemini are fairly stable under prompt rewording (CV ~0.38), but the local Qwen2.5-7B is not (CV ~0.75, 85% of records unstable) -- worth knowing before picking a fallback LLM baseline | `week4_consistency.md` |

## Repo map

```
data/            raw (gitignored) / processed (gitignored) / splits (committed) / samples (committed)
src/memorylife/  the installable package -- data, encoders, heads, losses, evaluation, utils
                 (features/, fusion/, memory/, retrieval/, inference/ are Week 5 stubs)
baselines/       llm_prompted_ttl.py, chatgpt_prompted_ttl.py, gemini_prompted_ttl.py,
                 bucket_classifier.py, heuristic_ttl.py (implemented);
                 fifo/lru/mem0/memgpt/... (Week-5 stubs)
scripts/         thin CLIs: preprocess.py, train.py, run_baseline.py, evaluate.py,
                 run_seed_sweep.py, compute_significance.py, compute_richer_metrics.py,
                 consistency_check.py, run_ablations.py, run_all.sh, run_smoke.sh
configs/         documented run parameters (not yet Hydra-wired, see docs/reproducibility.md)
experiments/     main/ (5-seed sweep), ablation/ (encoder + hyperparameter) -- resolved
                 config + checkpoint + metrics.json + log per run
results/         tables/ (committed), raw/ (gitignored)
docs/            architecture.md, benchmark_card.md, reproducibility.md, figures/
tests/           test_censoring.py, test_survival_loss.py (real coverage);
                 test_schema.py, test_features.py, ... (still empty stubs)
```

Full recommended tree and checklist mapping: `docs/repo_structure_reference.md`.

## Citation

See `CITATION.cff` (placeholder -- authors/DOI to be filled in for Week 6).

## License

Code: MIT (`LICENSE`, placeholder copyright holder -- confirm before
publishing). Data: unresolved, see `LICENSE-DATA`.
