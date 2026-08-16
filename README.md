# MemoryLifeBench

Memory lifetime prediction as a **time-to-event problem**. Instead of
asking "should I store this memory?", we ask "how long should I keep this
memory?" -- and train a survival model, not a classifier, to answer it.

Architecture figure: [`docs/figures/architecture.pdf`](docs/figures/architecture.pdf)
(GitHub renders PDFs on click; there's no PNG/SVG export yet -- see
`docs/architecture.md` for a box-by-box walkthrough of what's built vs.
still a stub).

**Status: Week 6 of 6, in progress.** Dataset (W1), benchmark (W2), the
first trained survival model + baseline comparison (W3),
multi-seed/significance/ablation/consistency experiments (W4), and the
full joint multi-task model + memory system + retrieval + grounded-QA
pipeline (W5) are done. Week 6 asks the real downstream question -- does
our forgetting policy actually help QA accuracy, not just rank memory
lifetime well -- and found and fixed a genuine problem along the way (see
below). The paper write-up itself is not started -- see
`docs/reproducibility.md` "Known gaps" for the full honest list before
assuming more of this repo works than actually does.

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

Week-5 joint model + full pipeline demo, also separate from `run_all.sh`:

```bash
python scripts/train_joint.py --fusion concat --seed 42   # ~1-3 min, produces artifacts/joint_model.pt
OPENROUTER_API_KEY=... python scripts/run_inference_demo.py --device cuda
```

The demo ingests a real 10-memory conversation (containing genuinely
conflicting facts on purpose), runs compaction/forgetting/reflection, then
answers 4 questions grounded in whatever survives, via a real GPT-4o call.

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

## Week-5: the full joint model + memory system

Adds auxiliary features (6 off-the-shelf pretrained extractors: temporal,
novelty, entities, intent, emotion, contradiction) fused with the embedding,
feeding THREE jointly trained heads (Lifetime, Action, Future-utility)
sharing one representation -- plus a 4th, heuristic-only Importance "head"
(no ground-truth label exists for it, see `src/memorylife/heads/importance.py`).

| Fusion | Test C-index | Test action accuracy | Test utility accuracy |
|---|---|---|---|
| **concat** | **0.7553 +/- 0.0045** | **0.8642 +/- 0.0005** | 0.6170 +/- 0.0072 |
| gated | 0.7304 +/- 0.0082 | 0.8384 +/- 0.0070 | 0.6165 +/- 0.0044 |

(3 seeds each.) Concat fusion beats the Week-3/4 lone survival head
(0.7312 +/- 0.0131 test) by a real, consistent margin -- and beats gated
fusion despite gated being the more sophisticated "ours" mechanism per the
architecture figure. Full table: `results/tables/week5_joint_model_results.md`.

On top of the joint model: a real memory store (in-memory, brute-force
cosine search -- sufficient at ~10K memories), a forgetting policy (TTL
expiry + Action-head "forget"), compaction (merges near-duplicates),
reflection (importance/utility decay past predicted TTL), an append-only
audit log, a retriever (similarity + importance + utility reranking), and
grounded QA via a real LLM call. `scripts/run_inference_demo.py` runs all
of this end-to-end and is honest about where it currently breaks: with two
conflicting memories at near-identical retrieval scores and no timestamp
reasoning in the prompt, the LLM flags the ambiguity rather than guessing
wrong -- a real limitation, documented in `docs/reproducibility.md`, not
hidden.

## Week-6: does the ranking win actually help downstream QA? (in progress)

Weeks 3-5 validated our lifetime model with C-index -- a metric that only
checks whether the model RANKS relative memory lifetimes correctly, never
whether any specific number it outputs is a safe absolute cutoff. Week 6
asks the real question: when our forgetting policy (Lifetime-head TTL
expiry + Action-head "forget") decides what to evict, does that actually
preserve QA-answering accuracy better than naive alternatives, at the same
storage budget? Tested on two real benchmarks (LoCoMo, LongMemEval) with
real GPT-4o answering, via `scripts/run_downstream_qa_eval.py`, against
three baselines: `no_forget` (ceiling), `fifo`, `lru`.

**First result: no.** At the original settings, `ours` scored *worst* of
all four on both benchmarks' EM/F1 (`results/tables/week6_downstream_qa.md`).

**Root-caused it** with a free, LLM-call-free diagnostic
(`scripts/diagnose_eviction_evidence.py`) that checks directly whether
each QA pair's gold-evidence memory survives eviction under each policy:

| Policy | Evidence retention (1,304 covered LoCoMo QA pairs) |
|---|---|
| no_forget | 100.0% |
| lru | 76.8% |
| fifo | 72.5% |
| **ours** | **66.9%** |

Broke the `ours` evictions of real evidence down by mechanism: **100% TTL
expiry, 0% Action-head "forget"** -- the action head's known per-class
precision gap (Week 5) is NOT the cause. The actual cause: `predicted_ttl_days`
was defined as the survival curve's **median** crossing (`S(t) = 0.5`).
Using a median as a hard cutoff is, by construction, a coin-flip threshold
-- roughly half of records that need to last longer than their own median
will, by definition, still be needed past it. Measured directly: evicted
evidence memories had a median predicted TTL of 99.5 days but were needed
to a median age of 162.9 days (`results/tables/week6_evidence_retention.md`).

**The fix**: made the cutoff quantile configurable (`quantile_ttl_days`,
`--ttl-quantile`) instead of hardcoding the median. A free sweep across
quantiles (`results/tables/week6_ttl_quantile_sweep.md`) shows evidence
retention climbs monotonically as the cutoff moves later (Q=0.5: 66.9% ->
Q=0.2: 90.8% -> Q=0.1: 95.0%), though it doesn't fully close the gap to
`fifo`/`lru` at any quantile tested. Confirmed on real EM/F1 with a
matched, same-145-question controlled comparison (`results/tables/week6_downstream_qa_q0.5_pilot_control.md`
vs. `..._q0.2_pilot.md` vs. `..._q0.1_pilot.md`): `ours` improved more than
`fifo` or `lru` did across the same quantile sweep, closing 61% of its
gap to the `no_forget` ceiling on LoCoMo F1 by Q=0.1, with no reversal yet
observed. LongMemEval showed no effect either way, as expected (its
conversations are too small, ~7 memories each, for a cutoff choice to
matter).

**Follow-up -- and this is the actual fix**: `ours` still didn't out-rank
`fifo`/`lru` after the quantile fix alone, because `ours` makes an
independent per-memory threshold decision while `fifo`/`lru` always keep
a ranked top-N. Added `ours_utility`: rank all memories by the
already-trained Future-Utility head's `P(retrieved again)` (AUC
0.71-0.77, previously used only to rerank retrieval results, never for
eviction) and keep the top-N, same structure as `fifo`/`lru`. Free
evidence-retention result (n=1,304 covered QA pairs -- large enough that
this is a robust proxy-metric margin): `ours_utility` beats `lru` at
every quantile tested, including at the ORIGINAL unfixed Q=0.5 (0.8765
vs. lru's 0.7676, vs. the original `ours`'s 0.6687) -- see
`results/tables/week6_ranked_eviction_sweep.md`.

**On real EM/F1** (`results/tables/week6_downstream_qa_q0.2_ranked_pilot.md`,
same matched 145-question sample): `ours_utility` scores EM 0.0917 / F1
0.1949 on LoCoMo, essentially tying the `no_forget` ceiling (EM 0.0917 /
F1 0.1957) while using only 91% of its storage. **Bootstrap-tested this
time** (`results/tables/week6_downstream_significance.md`, paired,
10,000 resamples, n=120 LoCoMo questions) rather than just reported as a
raw mean, per a reviewer gap this repo previously had (Week 3/4's
C-index claims got this treatment, Week 6's EM/F1 claims initially
didn't): `ours_utility` significantly beats both `fifo` (p=0.002 EM /
p=0.001 F1) and the original `ours` (p=0.006 EM / p=0.001 F1), and is
**statistically indistinguishable from the `no_forget` ceiling**
(EM diff exactly 0 on every one of 10,000 resamples -- it got the
identical set of questions right; F1 diff -0.0008, CI comfortably
straddling 0). The vs.-`lru` EM/F1 comparison, however, is **not**
significant at this sample size (p=0.367 EM / p=0.648 F1) -- "matches
the ceiling" is the well-supported headline claim here, not "beats lru,"
which the small paid sample can't yet distinguish from noise. The
TTL-quantile fix (Fix #1) is independently confirmed significant on F1
(Q0.2 vs Q0.5: p=0.010; Q0.1 vs Q0.5: p=0.005) though not yet on EM at
this sample size (p=0.137 / p=0.048).

**Why does this actually work?** (`scripts/analyze_utility_signal.py`,
`results/tables/week6_utility_signal_auc.md`, free, no LLM calls): pooled
AUC of each signal directly predicting "is this memory ever cited as QA
evidence" across all 2,536 LoCoMo memories. `utility_prob` scores 0.6709
-- genuinely predictive, and positive in every one of the 10
conversations individually (range 0.61-0.76), matching the Future-Utility
head's own Week-5 validation AUC. `predicted_ttl_days` scores **0.2852 --
below 0.5, i.e. INVERSELY correlated** with being QA evidence, not just
weak. A memory predicted to survive longer is actually LESS likely to be
what a question needs -- "how long until this fact goes stale" (the
survival objective) and "will this be the evidence for a specific
question" (what downstream QA needs) are different, sometimes inversely
related constructs; LoCoMo often asks about specific one-off events
(naturally short predicted lifetime) rather than durable facts. This
composes with the Fix #1 finding into one coherent causal story: the
original policy's median cutoff was a coin-flip AND, independent of any
cutoff choice, the underlying ranking signal points backwards relative to
what downstream QA needs -- which is why moving the cutoff (Fix #1) only
partly helped, and switching the ranking signal entirely (Fix #2) was
what actually closed the gap.

## Repo map

```
data/            raw (gitignored) / processed (gitignored) / splits (committed) / samples (committed)
src/memorylife/  the installable package -- data, encoders, features, fusion, heads,
                 losses, models, memory, retrieval, inference, evaluation, utils
                 (fusion/cross_attention.py, memory/store/{faiss,chroma}_store.py,
                 memory/store/sqlite_metadata.py are still stubs)
baselines/       llm_prompted_ttl.py, chatgpt_prompted_ttl.py, gemini_prompted_ttl.py,
                 bucket_classifier.py, heuristic_ttl.py, fifo.py, lru.py, no_forget.py
                 (implemented, see baselines/README.md); mem0/memgpt/
                 generative_agents_importance/locomo/longmemeval wrappers still stubs
scripts/         thin CLIs: preprocess.py, train.py, run_baseline.py, evaluate.py,
                 run_seed_sweep.py, compute_significance.py, compute_richer_metrics.py,
                 consistency_check.py, run_ablations.py, train_joint.py,
                 run_inference_demo.py, build_benchmark.py (Week-6 evidence-coverage
                 check), run_downstream_qa_eval.py (Week-6 policy comparison),
                 diagnose_eviction_evidence.py (Week-6 root-cause diagnostic),
                 run_all.sh, run_smoke.sh
configs/         documented run parameters (not yet Hydra-wired, see docs/reproducibility.md)
experiments/     main/ (5-seed sweep), ablation/ (encoder + hyperparameter),
                 joint/ (Week-5 fusion x seed sweep) -- resolved config +
                 checkpoint + metrics.json + log per run
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
