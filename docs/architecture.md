# Architecture

Figure: `figures/architecture.pdf`. Boxes are colour-coded in the original:
grey = existing components (off-the-shelf models), green = learned modules
(ours), orange = lifecycle & audit (ours).

| Box | What it is | Status | Code |
|---|---|---|---|
| Sentence Encoder (E5/BGE) | 768-d embedding of the input statement | **Built** | `src/memorylife/encoders/bge.py` |
| Semantic Feature Extractors (Intent, Entities, Temporal, Emotion/Preference, Novelty, Contradiction) | Auxiliary features fused with the embedding | **Built** (Week 5), all off-the-shelf pretrained models, none fine-tuned | `src/memorylife/features/*.py` |
| Feature Fusion -> fused vector z | Combines embedding + features | **Built** (Week 5): `concat` and `gated` variants; `cross_attention` still a stub | `src/memorylife/fusion/*.py` |
| Joint Lifecycle Predictor: **Lifetime head** (hazard h(t\|z) -> S(t\|z), TTL) | Survival model | **Built** (Week 3); also present inside the Week-5 joint model as an ablation, see `results/tables/week5_joint_model_results.md` | `src/memorylife/heads/survival.py`, `src/memorylife/losses/cox_partial.py` |
| Joint Lifecycle Predictor: Importance head | Score in [0,1] | **Heuristic only** (Week 5) -- no ground-truth importance label exists anywhere in the dataset schema, so this is a documented hand-written function, NOT a trained/learned head. See the module docstring before citing this as "learned importance." | `src/memorylife/heads/importance.py` |
| Joint Lifecycle Predictor: Future-utility head | P(retrieved in [t, t+delta]) | **Built** (Week 5), trained on the `observed_usage`/`no_usage_observed` subset (the only records with a genuine usage label) | `src/memorylife/heads/future_utility.py` |
| Joint Lifecycle Predictor: Action head | store/update/merge/forget | **Built** (Week 5), trained on labels derived from `lifecycle_event` | `src/memorylife/heads/action.py` |
| Memory Object | `{text, embedding, importance, TTL, type, action, provenance}` | **Built** (Week 5) | `src/memorylife/memory/memory_object.py` |
| Memory Store (vector DB + metadata), periodic reflection, self-compaction, forget+audit log | The lifecycle system | **Built** (Week 5): brute-force numpy vector store (sufficient at ~10K memories); `faiss_store.py`/`chroma_store.py`/`sqlite_metadata.py` remain stubs for when scale demands them | `src/memorylife/memory/*` |
| Retriever (sim(q,e) + importance + utility) | Downstream retrieval | **Built** (Week 5) | `src/memorylife/retrieval/*` |
| LLM -> grounded answer | Downstream QA | **Built** (Week 5): `scripts/run_inference_demo.py` runs the full pipeline end-to-end on a real conversation | `src/memorylife/inference/*` |
| Forgetting-policy comparison harness (`no_forget`/`fifo`/`lru`/`ours`/`ours_utility`/`ours_combo`, matched storage budget) | Does the Lifetime/Action-head forgetting policy actually preserve downstream QA quality vs. naive alternatives? | **Built** (Week 6): `ours_utility`/`ours_combo` rank-evict by the Future-Utility head instead of `ours`'s hard TTL-threshold cutoff | `scripts/run_downstream_qa_eval.py` |
| Configurable TTL cutoff (`quantile_ttl_days`, was hardcoded at the survival curve's median) | Fix #1 -- the median is a coin-flip cutoff by construction, not something C-index (rank-only) ever validates | **Built** (Week 6) | `src/memorylife/inference/pipeline.py` |
| Evidence-retention root-cause diagnostic (free, no LLM calls) | Isolates eviction-policy quality from LLM-answering noise via pure set membership against gold QA evidence | **Built** (Week 6), covers both LoCoMo and LongMemEval | `scripts/diagnose_eviction_evidence.py` |
| Downstream significance / mechanistic / judge-scoring / qualitative-tracing scripts | Bootstrap CI+p-value on the EM/F1 claims, AUC of `utility_prob`/`predicted_ttl_days` predicting QA-evidence relevance, LLM-judge rescoring, traced before/after examples | **Built** (Week 6) | `src/memorylife/evaluation/downstream_significance.py`, `scripts/analyze_utility_signal.py`, `scripts/judge_downstream_qa.py`, `scripts/qualitative_examples.py` |

## What Week 3 actually is, precisely

A single learned component: raw BGE embedding `e` -> small MLP -> log
partial-hazard, trained with the Cox partial-likelihood loss on
`(duration_days, event_observed)` pairs from MemoryLifeBench, correctly
handling right-censored records (`src/memorylife/data/censoring.py`).
Evaluated by concordance index against three baselines
(`baselines/`). No feature fusion, no other heads, no memory store, no
retrieval loop yet.

## What Week 5 adds, precisely

Auxiliary features (6 off-the-shelf pretrained extractors) fused with the
embedding via a learned gate or plain concatenation, feeding THREE jointly
trained heads (Lifetime, Action, Future-utility) sharing one fused
representation `z` -- trained with a custom loop (not pycox's `CoxPH.fit()`
wrapper, which can't share gradients across heads; see
`src/memorylife/models/multitask.py`'s docstring). The 4th head
(Importance) is a documented heuristic, not learned -- no ground-truth
label exists for it in this dataset.

Concat fusion + features + multi-task supervision reaches **0.7553 +/-
0.0045 test C-index** (3 seeds), a real improvement over the Week-3/4 lone
survival head's **0.7312 +/- 0.0131**. Concat beat gated fusion here,
consistently -- worth noting since gated is the more sophisticated
mechanism, not vindicated by the data on this dataset size. Full comparison:
`results/tables/week5_joint_model_results.md`.

On top of the joint model: a real memory store (`src/memorylife/memory/`),
forgetting policy (Lifetime-head TTL expiry + Action-head "forget"
predictions), compaction (merges near-duplicate memories), reflection
(importance/utility decay for memories past their predicted TTL), an
append-only audit log, a retriever (similarity + importance + utility
reranking), and a grounded-QA pipeline calling a real LLM (GPT-4o via
OpenRouter) over retrieved memories -- `scripts/run_inference_demo.py` runs
all of this end-to-end on a real MemoryLifeBench conversation containing
genuinely conflicting facts (two different phone numbers, two different
cities), and the demo output is honest about where it breaks: given two
memories with near-identical retrieval scores and no timestamp reasoning in
the prompt, the LLM correctly flagged the ambiguity rather than guessing
wrong with false confidence -- a real, documented limitation, not
papered over. See `docs/reproducibility.md`'s known-gaps section for what's
still not built (retrieval-scoring ablation, timestamp-aware disambiguation,
FAISS/Chroma backends, feature-ablation configs).

## What Week 6 adds, precisely

Everything above was validated on the model's OWN metrics (C-index,
action-accuracy). Week 6 asks the harder question: does the actual
forgetting policy (Lifetime-head TTL expiry + Action-head "forget") help
DOWNSTREAM QA answering, at a storage budget matched against naive
`fifo`/`lru` alternatives? Full write-up: `README.md`'s Week-6 section;
this is a pointer, not a duplicate.

Surprise negative result: `ours` LOST to both naive baselines on real
LoCoMo/LongMemEval EM/F1. Root-caused with a free (no-LLM-cost) evidence-
retention diagnostic (`scripts/diagnose_eviction_evidence.py`) down to two
independent causes: (1) `predicted_ttl_days` was defined as the survival
curve's MEDIAN -- a coin-flip cutoff by construction, never validated by
C-index since C-index is rank-only and scale-invariant; (2) `ours` makes an
independent per-memory threshold decision where `fifo`/`lru` rank-select
top-N, so a miscalibrated absolute cutoff has nowhere to hide.

Two fixes, both confirmed on real paid EM/F1 runs with bootstrap
significance testing (`scripts/compute_downstream_significance.py`), not
just the free diagnostic proxy:
- **Fix #1** (quantile cutoff, `quantile_ttl_days`): lower quantiles push
  the cutoff later, trading storage for retention -- see
  `results/tables/week6_ttl_quantile_sweep.md`.
- **Fix #2** (`ours_utility`: rank-evict by the Future-Utility head's
  `utility_prob` instead of a hard TTL threshold): reaches the `no_forget`
  ceiling and significantly beats `fifo`/original `ours` (bootstrap
  p<0.01); NOT statistically distinguishable from `lru` at n=120 -- stated
  as an honest limit, not oversold as "beats lru outright."
  `scripts/analyze_utility_signal.py` explains why mechanistically:
  `utility_prob` AUC=0.67 predicting real QA-evidence relevance,
  `predicted_ttl_days` AUC=0.29 (i.e. INVERSELY correlated) -- Fix #2 works
  because it replaces a backwards-ranking signal with a genuinely
  predictive one, not just because ranking beats thresholding in general.

Also: `scripts/judge_downstream_qa.py` found EM undercounts real answer
quality by ~22% (LLM-judge scoring); `scripts/qualitative_examples.py`
traced 8 concrete evicted-evidence-memory -> wrong-answer -> fixed-by-
keeping-it examples; the evidence-retention diagnostic was extended to
LongMemEval after an earlier claim that it had no evidence linkage turned
out to be untested and wrong (see `docs/reproducibility.md`).
