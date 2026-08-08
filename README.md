# MemoryLifeBench

Memory lifetime prediction as a **time-to-event problem**. Instead of
asking "should I store this memory?", we ask "how long should I keep this
memory?" -- and train a survival model, not a classifier, to answer it.

Architecture figure: [`docs/figures/architecture.pdf`](docs/figures/architecture.pdf)
(GitHub renders PDFs on click; there's no PNG/SVG export yet -- see
`docs/architecture.md` for a box-by-box walkthrough of what's built vs.
still a stub).

**Status: Week 3 of 6.** Dataset (W1), benchmark (W2), and the first
trained survival model + baseline comparison (W3) are done. Weeks 4-6
(calibration/consistency/cost experiments, the end-to-end memory system,
the paper) are not started -- see `docs/reproducibility.md` "Known gaps"
for the full honest list before assuming more of this repo works than
actually does.

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

## Results (Week 3: our survival model vs. 3 baselines, by C-index)

| Split | Method | C-index |
|---|---|---|
| test | **Our survival model (CoxPH on BGE embeddings)** | **0.7218** |
| test | Day/week/permanent classifier | 0.6298 |
| test | LLM-prompted TTL (Qwen2.5-7B, local) | 0.5207 |
| test | Recency-frequency heuristic | 0.4753 |
| val | **Our survival model (CoxPH on BGE embeddings)** | **0.7134** |
| val | Day/week/permanent classifier | 0.6195 |
| val | LLM-prompted TTL (Qwen2.5-7B, local) | 0.5713 |
| val | Recency-frequency heuristic | 0.4849 |

Full table: `results/tables/week3_results_table.md`. C-index = 0.5 is
random; 1.0 is perfect ranking.

## Repo map

```
data/            raw (gitignored) / processed (gitignored) / splits (committed) / samples (committed)
src/memorylife/  the installable package -- data, encoders, heads, losses, evaluation, utils
                 (features/, fusion/, memory/, retrieval/, inference/ are Week 4-5 stubs)
baselines/       llm_prompted_ttl.py, bucket_classifier.py, heuristic_ttl.py (implemented);
                 fifo/lru/mem0/memgpt/... (Week-5 stubs)
scripts/         thin CLIs: preprocess.py, train.py, run_baseline.py, evaluate.py, run_all.sh, run_smoke.sh
configs/         documented run parameters (not yet Hydra-wired, see docs/reproducibility.md)
results/         tables/ (committed), raw/ (gitignored)
docs/            architecture.md, benchmark_card.md, reproducibility.md, figures/
tests/           empty stubs -- no test coverage yet
```

Full recommended tree and checklist mapping: `docs/repo_structure_reference.md`.

## Citation

See `CITATION.cff` (placeholder -- authors/DOI to be filled in for Week 6).

## License

Code: MIT (`LICENSE`, placeholder copyright holder -- confirm before
publishing). Data: unresolved, see `LICENSE-DATA`.
