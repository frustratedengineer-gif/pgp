# Full comparison: population, time, parameters, tokens, accuracy (Weeks 3-5)

All numbers below are measured, not estimated, unless explicitly marked
"not metered" -- see the source table/log cited in each section. Wall-clock
times are single-GPU (shared 8x H200 node, see `docs/reproducibility.md`).

## Population (N records)

| Split | N | Notes |
|---|---|---|
| train | 8,297 | used for fitting all trained models |
| val | 939 | early stopping + Week-4/5 ablation selection |
| test | 916 | held-out, all headline numbers below are on this split |
| **total** | **10,152** | |

## Parameter counts: what we actually trained vs. what's frozen/off-the-shelf

This is the core of the "NLP models, not LLM" comparison: everything we
TRAIN is a small MLP on top of FROZEN pretrained encoders (never
fine-tuned). None of it is autoregressive generation.

| Component | Parameters | Trained by us? | Notes |
|---|---|---|---|
| Week-3/4 survival head | 213,889 | **Yes** | `heads/survival.py`, MLP on 768-d BGE embedding |
| Week-5 joint model (concat), total | 425,734 | **Yes** | fusion (0, concat has no params) + survival_net (221,057) + action_net (102,532) + utility_net (102,145) |
| BGE-base encoder | 109,482,240 | No (frozen) | `encoders/bge.py`, single forward pass per memory, embedding only |
| NER (dslim/bert-base-NER) | 107,726,601 | No (frozen) | `features/entities.py` |
| Intent zero-shot (distilbert-mnli) | 66,955,779 | No (frozen) | `features/intent.py` |
| Emotion (distilroberta) | 82,123,783 | No (frozen) | `features/emotion.py` |
| Contradiction NLI (distilbert-mnli) | 66,955,779 | No (frozen) | `features/contradiction.py`, same architecture as intent, separate instance |
| **Frozen backbone total** | **~433.2M** | No | all single-forward-pass classification/embedding models, zero autoregressive generation |
| Local Qwen2.5-7B-Instruct | ~7,000,000,000 | No (baseline, not ours) | `baselines/llm_prompted_ttl.py`, autoregressive generation |
| GPT-4o / Gemini 2.5 Pro | undisclosed (frontier-scale) | No (baseline, not ours) | `baselines/chatgpt_prompted_ttl.py` / `gemini_prompted_ttl.py`, autoregressive generation via API |

**What we actually learn (639,623 trainable params across both trained
models) sits on top of ~433M frozen, off-the-shelf, non-generative
parameters -- none of which involve token-by-token decoding.** The three
LLM baselines are 3-4+ orders of magnitude larger and, unlike our pipeline,
generate output token-by-token.

## Wall-clock time (single GPU, measured)

| Step | Time | Source |
|---|---|---|
| Feature pipeline: 6 extractors, all 10,152 records (one-time, cached after) | 162.6s | measured this session |
| Survival head training (Week 3/4, one seed) | 33.71s | measured this session, `/usr/bin/time -v` |
| Joint model training (Week 5, one seed/fusion) | 17.24s | measured this session, `/usr/bin/time -v` |
| `heuristic_ttl` baseline (all splits) | <5s | `docs/reproducibility.md` |
| `bucket_classifier` baseline (fit + predict all splits) | <5s | `docs/reproducibility.md` |
| Local Qwen2.5-7B baseline, val (939 records, 6 workers) | 86s | `docs/reproducibility.md` |
| Local Qwen2.5-7B baseline, test (916 records) | 78s | `docs/reproducibility.md` |
| GPT-4o baseline, val (939 records, 4 workers) | 319.5s | `results/tables/llm_token_usage.md` |
| GPT-4o baseline, test (916 records) | 287.2s | `results/tables/llm_token_usage.md` |
| Gemini 2.5 Pro baseline, val (939 records, 4 workers) | 1488.2s | `results/tables/llm_token_usage.md` |
| Gemini 2.5 Pro baseline, test (916 records) | 1473.7s | `results/tables/llm_token_usage.md` |

Our own training steps (survival head, joint model) run end-to-end,
including data loading and evaluation, in under 35 seconds each. The
frontier-LLM baselines take 5-25 minutes PER SPLIT for the exact same
~900-1000 records, because every record is a separate network round-trip
plus autoregressive generation, not one batched forward pass.

## Token usage

| Method | Val tokens | Test tokens | Val+test total | Metered/billed? |
|---|---|---|---|---|
| GPT-4o | 130,838 | 127,947 | 258,785 | Yes (real OpenRouter API spend) |
| Gemini 2.5 Pro | 108,227 | 101,845 | 210,072 | Yes (real OpenRouter API spend) |
| Local Qwen2.5-7B | not metered | not metered | not metered | No (self-hosted, no per-token billing) -- still a 7B-parameter autoregressive model doing real token-by-token generation per call, just not instrumented to log counts in `baselines/llm_prompted_ttl.py` |
| **Our survival head / joint model / feature pipeline** | **0** | **0** | **0** | **N/A -- not applicable.** None of these are autoregressive text generation. BGE encoding is one forward pass producing a fixed 768-d vector; the 4 feature-extractor pipelines are one forward pass each producing classification probabilities; the survival/action/utility heads are single forward passes through a small MLP. There is no token-by-token decoding anywhere in our pipeline, so "tokens generated" is a metric that doesn't exist for it -- not a small number, a **structurally absent one**. |

This is the concrete form of "we use NLP models, not an LLM, so token
spend is lower": it's not lower, it's a different computational paradigm
that the token-cost metric doesn't apply to at all. The trade-off is real
and worth stating plainly: those ~433M frozen parameters and one BGE
forward pass still cost GPU time and memory (see param table above) -- the
comparison is fair and favorable to us, but not "free."

## Accuracy: test-split C-index / action-accuracy / utility-accuracy

| Method | Test C-index | Test action acc | Test utility acc | Source |
|---|---|---|---|---|
| Recency-frequency heuristic | 0.4753 | -- | -- | `week3_results_table.md` |
| Gemini 2.5 Pro (zero-shot) | 0.4806 | -- | -- | `week3_results_table.md` |
| Local Qwen2.5-7B (zero-shot) | 0.5207 | -- | -- | `week3_results_table.md` |
| GPT-4o (zero-shot) | 0.5411 | -- | -- | `week3_results_table.md` |
| Day/week/permanent classifier | 0.6298 | -- | -- | `week3_results_table.md` |
| **Our survival head (Week 3/4, single seed=42)** | **0.7218** | -- | -- | `week3_results_table.md` |
| **Our survival head (Week 3/4, 5-seed mean +/- std)** | **0.7312 +/- 0.0131** | -- | -- | `week4_multiseed_results.md` |
| **Our joint model, gated fusion (Week 5, 3-seed mean +/- std)** | **0.7304 +/- 0.0082** | **0.8384 +/- 0.0070** | 0.6165 +/- 0.0044 | `week5_joint_model_results.md` |
| **Our joint model, concat fusion (Week 5, 3-seed mean +/- std)** | **0.7553 +/- 0.0045** | **0.8642 +/- 0.0005** | 0.6170 +/- 0.0072 | `week5_joint_model_results.md` |

Every "our model" row beats every LLM/heuristic baseline row, with the
comparison statistically significant at p<0.001 for every baseline
(`week4_significance.md`, bootstrap, 1000 resamples).

## Per-class detail the aggregate accuracy numbers above hide

Action-head aggregate accuracy (~86%) is driven heavily by the majority
"store" class. Per-class breakdown (concat fusion, seed 42, test split;
full detail across all 6 checkpoints in `week5_action_utility_detail.json`):

| Action class | Precision | Recall | F1 | Support (test) |
|---|---|---|---|---|
| store | 1.000 | 0.839 | 0.912 | 775 |
| update | 0.550 | 1.000 | 0.710 | 60 |
| merge | 0.457 | 1.000 | 0.627 | 48 |
| forget | 0.635 | 1.000 | 0.776 | 33 |

**This pattern (precision 0.41-0.69 across all 6 checkpoints -- the
single-checkpoint table above shows 0.46-0.64 for its own 3 rows --
recall exactly 1.000) holds across every single one of the 6 checkpoints
(2 fusion variants x 3 seeds)** --
consistent enough that it's a real property of training under aggressive
inverse-frequency class weighting (`losses/action_loss.py`), not
seed-specific noise. The model never MISSES a true update/merge/forget
case, at the cost of over-flagging some "store" records as needing action.
For a memory system, this is a defensible operating point (silently
keeping a stale/wrong fact is worse than an extra review flag) -- but it
means the ~86% aggregate accuracy number alone overstates how clean the
head's behavior is per-class, and should not be quoted without this table
next to it.

Future-utility head: AUC 0.71-0.77 across all 6 checkpoints (see
`week5_action_utility_detail.json`) -- genuinely predictive, well above the
0.5 random baseline, but not highly accurate; consistent with this being a
harder, noisier binary target than the others (see
`heads/future_utility.py`'s docstring on label quality).

## Verification performed this pass (not just re-asserted)

- **Causal-leakage proof**: `tests/test_causal_features.py` (5 new tests,
  all passing) formally proves `novelty.py`/`contradiction.py`'s shared
  `nearest_prior_in_conversation` helper never compares a record against a
  later one, and never crosses conversation boundaries -- previously only
  claimed in a docstring.
- **Regression check**: the causal-logic refactor (deduplicating
  novelty.py/contradiction.py into one shared, tested function) produces
  numerically identical features to the pre-refactor version (max abs diff
  2.4e-7, float32 noise floor) -- verified by direct comparison, not
  assumed.
- **Label completeness on ALL splits**, not just train: 0 unmapped
  `lifecycle_event` values across train (8,297), val (939), and test (916).
  Utility-label coverage: 61.4% train / 72.7% val / 71.1% test, positive
  class 29-34% of the labeled subset in every split -- checked directly,
  not assumed from the train-split check alone.
- **Found and fixed a real bug** in `scripts/eval_joint_detail.py`
  (float-vs-int label dtype mismatch broke `classification_report`'s dict
  keys) while producing the per-class table above.
- **Re-audited** `compute_significance.py`, `compute_richer_metrics.py`,
  `run_ablations.py`, `consistency_check.py` for the same class of bug that
  was already found and fixed in the Week-4 encoder ablation (BGE-large
  test embeddings silently using the wrong model). No further correctness
  bugs found; one real (non-correctness) inefficiency noted:
  `consistency_check.py`'s cache namespace doesn't overlap with the main
  baseline runs', so its first paraphrase (identical to the base prompt)
  re-queried instead of reusing already-paid-for API calls.
