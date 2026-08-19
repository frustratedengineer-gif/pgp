# MemoryLifeBench: Memory Lifetime Prediction as a Time-to-Event Problem

Bhargav Shendge

## Abstract

Personal AI assistants that accumulate memories over long-running conversations eventually face a storage problem: which memories should be kept, and for how long? Prior memory-management systems for LLM agents typically answer this with a binary or heuristic decision ("store or don't," "forget after N days," recency/frequency scores). We reframe the problem as **survival analysis**: instead of predicting whether a memory matters, we predict *how long it will remain useful*, using the standard time-to-event machinery (right-censored durations, concordance index) from clinical and reliability statistics. A small model (213,889 parameters, a single MLP on top of a frozen sentence embedding) trained with a Cox partial-likelihood loss beats three LLM-prompted baselines (GPT-4o, Gemini 2.5 Pro, a locally hosted Qwen2.5-7B) and two heuristic baselines on concordance index by a wide, statistically significant margin (p<0.001 against every baseline, bootstrap). Extending this to a joint multi-task model (Lifetime, Action, Future-Utility heads sharing one fused representation, still under 500K trainable parameters) improves ranking quality further and adds usable action/utility signals. But a good ranking metric is not the same as a good deployment policy: when we tested whether our learned forgetting policy actually preserves downstream question-answering quality at a matched storage budget against naive first-in-first-out and least-recently-used baselines, our original policy *lost* to both. We diagnose this failure to two root causes — a miscalibrated absolute threshold (the median of the survival curve is a coin-flip cutoff by construction) and a poor decision *structure* (independent per-memory thresholding instead of ranked top-N selection) — and show that fixing the second one closes almost the entire gap: ranking memories by the already-trained Future-Utility head's score, instead of thresholding the Lifetime head's TTL estimate, reaches the same downstream accuracy as a no-forgetting ceiling and significantly beats the original policy, while remaining statistically indistinguishable from LRU on exact-match/F1 at our current sample size. We report this result honestly, including where our fix does *not* clear statistical significance, and provide a mechanistic explanation (AUC of each candidate ranking signal against real QA-evidence relevance) for why the fix works. We also show that the no-forgetting "ceiling" itself is not the true achievable ceiling: an oracle given only the correct evidence, with no retrieval noise, significantly outperforms no-forgetting too (p<0.01), meaning retrieval quality is a genuine bottleneck this paper's eviction-policy fixes do not address. Finally, against a genuinely independent memory system (Mem0), our best policy is statistically tied, not superior — reported plainly rather than reframed favorably.

## 1. Introduction

**Motivation.** Long-running personal AI assistants accumulate a growing store of extracted facts ("memories") about the user across sessions. Unbounded storage is not viable at scale, and unbounded context windows do not solve the problem either — retrieval quality degrades as the store grows, and stale or contradicted facts actively hurt answer quality if never removed. Existing memory-system designs (Generative Agents' importance scoring, MemGPT's paging, Mem0's heuristic decay) treat "should this memory be forgotten" as a scalar importance or recency score with no explicit temporal semantics: none of them predict *when* a memory will stop being useful.

**Our reframing.** We treat every memory as a survival-analysis subject: a statement is "born" when made and "dies" (becomes invalid, contradicted, superseded, or simply never referenced again) at some later time, possibly never observed within the data we have (right-censoring). This gives us access to a mature, well-understood statistical toolkit (Cox proportional hazards, concordance index, time-dependent AUC, integrated Brier score) instead of ad hoc importance heuristics, and lets us evaluate "how good is this lifetime prediction" independent of any one downstream task.

**Contributions.**
1. **MemoryLifeBench**, a 10,152-record benchmark of memory statements with time-to-event labels, combining synthetic dialogues with exact provable lifetimes and real conversations extracted from two published long-term-memory benchmarks (LoCoMo, LongMemEval), split by conversation to avoid leakage (Section 3).
2. A lightweight survival model (a single MLP on frozen sentence embeddings) that beats three frontier/local LLM baselines and two heuristics at ranking memory lifetimes, with a full multi-seed, multi-metric, statistically tested evaluation (Section 5).
3. A joint multi-task extension (Lifetime + Action + Future-Utility heads sharing one representation) that improves on the single-task model and adds two more usable signals for a real memory system (Section 5.3).
4. A rigorous, honestly-reported test of whether any of this actually helps: a matched-storage-budget downstream QA comparison against naive forgetting policies, a negative initial result, a root-cause diagnosis, two fixes, and a confirmation on real LLM-scored QA accuracy with bootstrap significance testing throughout (Section 6). This section also reports where our fix does *not* reach significance, rather than only the results that support the headline claim.

## 2. Related Work

**Heuristic-scoring memory systems.** Generative Agents (Park et al., 2023) scores memories by a hand-tuned recency/importance/relevance blend, decayed exponentially; MemGPT (Packer et al., 2023) treats context as an OS-style paging problem between a fixed working set and external storage, not a lifetime-prediction problem; Mem0 (Chhikara et al., 2025) extracts and consolidates conversational memory with a heuristic update mechanism, frequently discarding extracted statements by its own internal decision rather than a calibrated forgetting policy (an observation also made independently by Shu et al., 2026). Unlike the other systems in this section, Mem0 is empirically compared against our own policies directly, not just discussed (Section 6.15). MemoryBank (Zhong et al., 2024) applies time-decay-based consolidation and forgetting directly analogous in spirit to our Lifetime head, but as a fixed decay curve rather than a per-memory learned prediction; Reflective Memory Management (Tan et al., 2025) further refines what to store/retrieve over time via LLM-driven reflection rather than a trained model. None of these systems predicts an explicit time-to-event target with an evaluable ranking metric, and to our knowledge none evaluates its forgetting policy against a matched-storage-budget naive baseline the way Section 6 does here.

**Structure-augmented (graph-based) memory systems.** HippoRAG (Gutierrez et al., 2024) and HippoRAG 2 (Gutierrez et al., 2025) organize knowledge into a graph for associative retrieval and continual updates; GraphRAG (Edge et al., 2024) builds LLM-summarized graphs for query-focused summarization; Zep/Graphiti (Rasmussen et al., 2025) maintains a temporal knowledge graph with context-assembly pipelines; A-Mem (Xu et al., 2025) performs agentic, Zettelkasten-style dynamic linking among notes. These systems improve retrieval structure over flat vector search, but generally do not model an explicit temporal dimension for individual memories, and none of them frame memory retention as a survival-analysis problem.

**Episodic memory and reasoning.** REMem (Shu et al., 2026) formalizes episodic memory as time-aware gists and facts in a hybrid graph, with an agentic tool-using retriever for multi-step temporal reasoning — evaluated on LoCoMo, REALTALK, Complex-TR, and Test of Time, beating Mem0/Graphiti/HippoRAG 2 by a wide margin and reporting refusal precision/recall/F1 on LoCoMo's adversarial questions, a methodology Section 6.11 adopts directly. REMem's own contribution is architectural (how to represent and retrieve episodic memory); ours is orthogonal — a learned, evaluable model of *when* a memory stops being useful, which could in principle sit underneath a system like REMem's own retrieval layer rather than compete with it. Several evaluation practices in Section 6 and Appendices A–C follow REMem's own methodology directly, credited at each point of use.

**Long-term conversational memory benchmarks.** LoCoMo (Maharana et al., 2024) and LongMemEval (Wu et al., 2024) are the two real-conversation benchmarks we build on; both provide multi-session dialogues with reference QA pairs but were designed to evaluate end-to-end memory systems' QA accuracy, not lifetime prediction in isolation — we repurpose their dialogue/evidence structure for MemoryLifeBench's censoring labels and, in Section 6, for a direct downstream evaluation.

**Survival analysis.** We use the standard Cox proportional hazards model (Cox, 1972) via `pycox`/`torchtuples` (Kvamme et al., 2019) and concordance index, integrated Brier score, and time-dependent AUC as evaluation metrics — standard practice in clinical/reliability survival modeling, not novel to this work; the contribution is applying this machinery to memory lifecycle prediction and carrying it through to a real downstream evaluation, including the Section 6.3 finding that a model can be an excellent ranker (high C-index) while being systematically miscalibrated on the absolute cutoff a deployed policy actually thresholds against — a distinction the clinical survival literature is well aware of (calibration vs. discrimination) but that, to our knowledge, prior LLM-agent memory-system papers do not discuss.

## 3. MemoryLifeBench: Problem Formulation and Dataset

### 3.1 Time-to-event framing

For every memory record we define a duration `T = duration_days` (time from when the statement was made to the reference event) and an event indicator `delta = event_observed` (1 if the event was actually observed, 0 if censored). A model predicts a hazard/survival function `S(t|z)` over an embedding `z` of the memory text; ranking quality is measured by the concordance index (C-index), which is rank-only and invariant to monotonic rescaling of the risk score — a property that becomes directly relevant to a real failure mode found in Section 6.

### 3.2 Data sources and composition

| Source | Train | Val | Test | What it is |
|---|---|---|---|---|
| Synthetic | 3,199 | 256 | 265 | LLM-generated multi-session dialogues with facts injected and probe questions scheduled by construction (exact, provable lifetimes) |
| LongMemEval | 3,162 | 359 | 375 | Candidate memories extracted from LongMemEval multi-session dialogues |
| LoCoMo | 1,936 | 324 | 276 | Candidate memories extracted from LoCoMo multi-session dialogues |
| **Total** | **8,297** | **939** | **916** | |

Splits are by conversation (568 / 71 / 71 conversations), verified disjoint — no conversation's memories leak across splits. Event rate: 37.2% train, 35.5% val, 36.6% test (by source, train: LoCoMo 48.8%, synthetic 43.1%, LongMemEval 24.1% — LongMemEval's low rate reflects that most of its records are never referenced again within the transcript, i.e. administratively censored rather than genuinely permanent).

### 3.3 Censoring convention

A memory is censored (no observed invalidation/update/contradiction/expiry within the data) under three cases: (1) event observed, T = invalidation time minus statement time; (2) censored with scheduled probes (synthetic data only), T = time of the last scheduled probe minus statement time; (3) censored with no probes (real conversations), T = the latest timestamp observed anywhere in that conversation minus statement time — an administrative-censoring proxy. Real conversations have a median of 8 memory records spanning a median 17.5 days (up to 298 days), so this proxy is meaningfully informative, not a single global cutoff. This is a judgment call, not a given fact — case 3 has no ground-truth alternative in the source data, and we flag it here rather than presenting it as settled.

## 4. Method

### 4.1 Survival model (single-task)

A single learned component: a frozen BGE-base (768-d) sentence embedding of the memory text feeds a small MLP (213,889 parameters) producing a log partial hazard, trained with the Cox partial-likelihood loss on (duration, event-observed) pairs, correctly handling right censoring. No feature fusion, no other heads, no memory store or retrieval loop at this stage. Formal notation (hazard/survival function, partial-likelihood objective, C-index) is given in Appendix A.1.

### 4.2 Joint multi-task model

Six off-the-shelf, frozen, non-fine-tuned auxiliary feature extractors (Intent, Entities/NER, Temporal, Emotion/Preference, Novelty, Contradiction — approximately 433M frozen parameters combined, a single forward pass each, zero autoregressive generation) are fused with the BGE embedding via either a learned gate or plain concatenation, feeding three jointly trained heads that share one fused representation:

- **Lifetime head**: the same Cox survival objective as in Section 4.1, now over the fused representation.
- **Action head**: 4-way classification (store/update/merge/forget), trained on labels derived from the record's lifecycle event, under aggressive inverse-frequency class weighting.
- **Future-Utility head**: binary P(retrieved again), trained only on the subset of records with a genuine usage label.

The model is trained with a custom loop rather than `pycox`'s single-task fitting wrapper, which cannot share gradients across heads. A fourth "Importance" score exists in the architecture but is a documented hand-written heuristic, not a learned head — no ground-truth importance label exists anywhere in the dataset schema, and we do not claim it is learned. Total trainable parameters for the concat-fusion joint model: 425,734 (concatenation itself adds zero parameters).

### 4.3 Memory system

Trained heads feed a real memory-lifecycle system: memory objects (`{text, embedding, importance, predicted_ttl_days, action, utility_prob, provenance}`) held in a brute-force numpy vector store (sufficient at MemoryLifeBench's ~10K-memory scale), with a forgetting policy (Lifetime-head TTL expiry plus Action-head "forget" predictions), compaction (merges near-duplicate memories), reflection (utility decay past predicted TTL), an append-only audit log, and a retriever that reranks similarity-matched candidates by importance and utility before a grounded-QA LLM call.

## 5. Experiments Part 1: Lifetime Prediction Quality

### 5.1 Setup and baselines

Five baselines, all evaluated on the same test split (N=916) and validation split (N=939): a recency-frequency heuristic, a day/week/permanent bucket classifier, and three LLM-prompted-TTL baselines (a locally hosted Qwen2.5-7B-Instruct; GPT-4o and Gemini 2.5 Pro via the OpenRouter API, zero-shot prompted for a TTL estimate; exact prompt in Appendix C.1). Real API spend for the two frontier baselines: 258,785 tokens (GPT-4o) and 210,072 tokens (Gemini), metered and billed. Our own pipeline has zero token cost by construction — no component performs autoregressive generation.

### 5.2 Headline results

| Split | Method | C-index |
|---|---|---|
| Test | **Our survival model** | **0.7218** |
| Test | Day/week/permanent classifier | 0.6298 |
| Test | LLM-prompted TTL (GPT-4o) | 0.5411 |
| Test | LLM-prompted TTL (Qwen2.5-7B) | 0.5207 |
| Test | LLM-prompted TTL (Gemini 2.5 Pro) | 0.4806 |
| Test | Recency-frequency heuristic | 0.4753 |

(Validation split shows the same ranking, our model at 0.7134.) Every "our model" row beats every baseline row with p<0.001 (one-sided bootstrap, 1,000 resamples) — e.g. against GPT-4o on test: +0.1801 C-index, 95% CI [+0.1340, +0.2235].

**Richer metrics** (time-dependent AUC, integrated Brier score) confirm the same ranking: our model reaches 0.7895 AUC / 0.2463 IBS (test) versus GPT-4o's 0.5339 AUC / 0.4882 IBS. Brier/IBS for the baselines uses a degenerate step-function survival curve built from each baseline's scalar point estimate — a legitimate way to score a point forecast under a proper scoring rule, but not evidence those baselines produce a calibrated probability curve the way our fitted Cox model does.

**Determinism.** Paraphrase-consistency testing (100 records × 3 paraphrases) shows our model has zero coefficient of variation (fully deterministic given fixed weights) while the LLM baselines vary substantially on semantically identical inputs — GPT-4o mean CV 0.3845 (44% of records with CV>0.5), Qwen2.5-7B mean CV 0.7456 (85%). This is a second, orthogonal argument for the approach beyond raw ranking accuracy: a deployed forgetting policy that changes its mind on a reworded version of the same fact is a liability regardless of its average accuracy.

### 5.3 Stability, ablations, and the joint model

**Five-seed stability** (seeds 13, 42, 1337, 2024, 7): 0.7312 ± 0.0131 test C-index, 0.7237 ± 0.0063 validation — low variance, not a lucky single-seed draw.

**Encoder ablation**: swapping BGE-base (768d) for BGE-large (1024d) gives 0.7382 ± 0.0048 versus 0.7312 ± 0.0131 test — a small, within-noise difference, not a strong argument for the larger encoder at this dataset size.

**Joint model** (concat fusion, 3 seeds): 0.7553 ± 0.0045 test C-index, a real, non-overlapping-error-bar improvement over the single-task head's 0.7312 ± 0.0131, achieved with a *tighter* seed spread. Concat fusion consistently beat gated fusion (0.7304 ± 0.0082) despite gated being the more parameter-rich mechanism — a result we do not over-explain, noting only that gated fusion may be more prone to overfitting at this dataset's ~8.3K-record training scale.

**Action head**: 86.4% aggregate test accuracy, but per-class detail (consistent across all 6 checkpoints, 2 fusion variants × 3 seeds) shows this hides a real pattern: precision ranges 0.41–0.69 on the minority update/merge/forget classes, with recall exactly 1.000 on all three across all 6 checkpoints. The inverse-frequency class weighting used in training means the model never *misses* a true update/merge/forget case, at the cost of over-flagging some "store" records — a defensible operating point for a memory system (a missed forget is worse than an extra review flag), but the aggregate 86% figure alone overstates how clean the per-class behavior is. This precision gap on the forget class becomes directly relevant to the downstream failure diagnosed in Section 6.3.

**Future-Utility head**: AUC 0.71–0.77 across all 6 checkpoints — genuinely predictive, well above the 0.5 random baseline, but noisier than the other two heads, consistent with a harder and more coarsely labeled binary target.

## 6. Experiments Part 2: Does It Help Downstream QA?

Section 5 validates ranking quality on the model's own metrics. This section asks the harder question: does the resulting forgetting policy actually preserve downstream question-answering quality, at a storage budget matched against naive alternatives?

### 6.1 Setup

Three forgetting policies are compared at the same final storage budget per conversation: `no_forget` (retain everything, an accuracy ceiling), `fifo` (keep the N most-recently-created memories), `lru` (keep the N most-recently-referenced, using a causal recency signal derived from a nearest-prior-in-conversation computation, not a live query stream), and `ours` (our original policy: Lifetime-head TTL expiry plus Action-head "forget"). `fifo`/`lru`'s capacity N is set to `ours`'s own natural final active count per conversation, so this is an apples-to-apples storage-budget comparison, not an arbitrary fixed budget. Formal definitions of all six policies (`no_forget`, `fifo`, `lru`, `ours`, `ours_utility`, `ours_combo`) are given in Appendix A.2. Evaluation uses real LoCoMo and LongMemEval questions, answered by GPT-4o over the retrieved memory store (exact prompt in Appendix C.2), scored by exact match (EM) and token-F1. LoCoMo's category-5 "adversarial" QA pairs (no true answer field, only a plausible-sounding wrong decoy) are excluded from this main comparison, since scoring against the decoy would reward hallucination; they are used separately in Section 6.11.

### 6.2 Surprise negative result

| Benchmark | Policy | Mean EM | Mean F1 | N |
|---|---|---|---|---|
| LoCoMo | no_forget | 0.1706 | 0.3007 | 1,542 |
| LoCoMo | fifo | 0.1316 | 0.2347 | 1,542 |
| LoCoMo | lru | 0.1245 | 0.2319 | 1,542 |
| LoCoMo | **ours** | **0.1102** | **0.2054** | 1,542 |

(LongMemEval shows the same ordering at smaller magnitude.) `ours` was the *worst* policy tested, below both naive baselines and well below the no-forgetting ceiling — the opposite of what Section 5's C-index results would predict.

### 6.3 Root-cause diagnosis

We built a free (no LLM calls) evidence-retention diagnostic: for every QA pair whose gold evidence memory was actually extracted, we check by pure set membership whether that memory survives each policy's eviction, isolating policy quality from LLM-answering noise entirely.

| Policy | Evidence retention rate (LoCoMo, N=1,304 covered QA pairs) |
|---|---|
| no_forget | 1.0000 |
| lru | 0.7676 |
| fifo | 0.7247 |
| **ours** | **0.6687** |

Two independent causes, diagnosed directly rather than assumed:

1. **Miscalibrated absolute threshold.** The predicted TTL was defined as the survival curve's *median* — a coin-flip cutoff by construction: by definition of median, roughly half of correctly ranked records are still "alive" past it. C-index is rank-only and scale-invariant, so it never validated this specific absolute cutoff choice. For evicted evidence memories, the mean predicted TTL was 114.4 days against a mean actual age needed of 164.1 days — a 49.7-day mean shortfall.
2. **Decision structure.** 100% of `ours`'s evicted-evidence-memory losses were attributable to TTL expiry, 0% to the Action head's "forget" prediction. `ours` makes an *independent per-memory threshold* decision where `fifo`/`lru` always keep their best-N by a *ranked* score — a threshold can waste capacity on a mediocre item whose score happens to land on the "keep" side while a genuinely important one with noisier scoring falls just short.

### 6.4 Fix #1: quantile cutoff

Making the TTL cutoff a configurable quantile instead of the hardcoded median lets us trade storage for retention. A free sweep confirms the fix direction:

| Q (S(t) cutoff) | `ours` storage kept | `ours` evidence retention |
|---|---|---|
| 0.5 (original) | 70.9% | 0.6687 |
| 0.2 | 91.3% | 0.9080 |
| 0.1 | 95.2% | 0.9502 |
| 0.05 | 97.1% | 0.9716 |

But this does *not* fully close the gap against `fifo`/`lru` at matched capacity at any quantile tested — `lru` retains evidence at least as well as `ours` at every single row (e.g. Q=0.2: `ours` 0.9080 vs. `lru` 0.9517) — confirming cause (1) is real but leaving cause (2), the decision structure, as a residual gap.

### 6.5 Fix #2: ranked utility eviction

`ours_utility` ranks all memories by the already-trained Future-Utility head's `utility_prob` (a signal previously used only to rerank retrieval results, never consulted by eviction), keeping the top-N matching `ours`'s own capacity — the same ranked-top-N *structure* `fifo`/`lru` use, substituting a trained score for a heuristic one. On the same free diagnostic:

| TTL quantile | fifo | lru | ours | ours_utility |
|---|---|---|---|---|
| 0.5 (original) | 0.7247 | 0.7676 | 0.6687 | **0.8765** |
| 0.2 | 0.9225 | 0.9517 | 0.9080 | **0.9663** |
| 0.1 | 0.9670 | 0.9709 | 0.9502 | **0.9877** |

`ours_utility` beats every baseline at every quantile tested, including at the original unfixed Q=0.5 — structure alone recovers most of the gap. A blended variant (`ours_combo`, 50/50 utility and continuous TTL signal) is a real improvement over `ours` but consistently underperforms pure `ours_utility` — the utility signal is doing essentially all of the work; blending dilutes rather than helps, echoing the "simpler was better" finding for fusion in Section 5.3.

### 6.6 Confirmation on real EM/F1, with significance testing

The free diagnostic is a proxy; we confirmed Fix #2 on a matched sample (n=120 LoCoMo questions, n=25 LongMemEval questions) of real GPT-4o-scored EM/F1, with paired bootstrap significance testing (10,000 resamples, same-question resampling) rather than reporting raw means:

| Comparison | N | EM diff, 95% CI | p (a beats b) |
|---|---|---|---|
| ours_utility vs. fifo | 120 | +0.0499 [+0.0167, +0.0917] | 0.002 |
| ours_utility vs. ours | 120 | +0.0415 [+0.0083, +0.0833] | 0.006 |
| ours_utility vs. no_forget | 120 | +0.0000 [+0.0000, +0.0000] | 1.000 |
| **ours_utility vs. lru** | 120 | +0.0083 [+0.0000, +0.0250] | **0.367** |

`ours_utility` significantly beats `fifo` and the original `ours` policy (p<0.01 on both metrics) and is statistically indistinguishable from the `no_forget` ceiling (EM difference exactly 0 across all 10,000 resamples) — though Section 6.14 shows `no_forget` itself is not the true achievable ceiling, so this claim should be read as "matches the best of the policies actually compared here," not "matches the limit of what's achievable." It is *not* significantly different from `lru` at this sample size (p=0.367 EM, p=0.648 F1) — the free-diagnostic sweep in Section 6.5 shows a large, consistent advantage over `lru` at every quantile, but we report the honest limit of what n=120 real LLM-scored questions can establish rather than the stronger claim the proxy metric alone would suggest.

As a cross-check against a metric family EM/F1 does not cover, we also computed BLEU-1 (unigram precision times brevity penalty) over these same predictions, a pure string metric recomputed post hoc at no additional cost:

| Benchmark | Policy | Mean EM | Mean F1 | Mean BLEU-1 | Mean Judge |
|---|---|---|---|---|---|
| LoCoMo | fifo | 0.0417 | 0.1294 | 0.0915 | 0.1833 |
| LoCoMo | lru | 0.0833 | 0.1998 | 0.1586 | 0.3417 |
| LoCoMo | no_forget | 0.0917 | 0.1957 | 0.1562 | 0.3333 |
| LoCoMo | ours | 0.0500 | 0.1575 | 0.1116 | 0.2000 |
| LoCoMo | **ours_utility** | 0.0917 | 0.1949 | **0.1549** | **0.3500** |

BLEU-1 preserves the same ranking as F1 and the judge score (`ours_utility` ≈ `no_forget` > `lru` > `ours` > `fifo`) — a confirmatory result across a fourth metric family, not a new finding on its own.

### 6.7 Why does the fix work? A mechanistic explanation

Beyond the outcome, we measured whether the two candidate ranking signals actually predict ground-truth QA-evidence relevance, independent of any eviction policy: pooled AUC of `utility_prob` and predicted TTL against a binary evidence label, across all 2,536 LoCoMo memories.

| Signal | Pooled AUC (predicts is_evidence) |
|---|---|
| utility_prob (Future-Utility head) | **0.6709** |
| predicted_ttl_days (Lifetime head, median quantile) | **0.2852** |

`utility_prob` is a genuinely predictive signal (positive in all 10 conversations individually, 0.61–0.76 range). Predicted TTL is *below* 0.5 — inversely correlated with evidence relevance. This gives a complete causal account of the Section 6.5 result: Fix #2 works not merely because ranking beats thresholding in general, but because it replaces a signal that is actively backwards with one that is genuinely predictive.

### 6.8 Does EM/F1 undercount real quality?

LLM-judge rescoring (GPT-4o judges each prediction against the reference, independent of exact wording; exact prompt in Appendix C.3) of the same 725 predictions found 22.1% (160/725) were marked wrong by EM but judged substantively correct. Under the judge metric, LoCoMo scores rise approximately 3.6–4.4x across every policy (e.g. `ours_utility`: EM 0.0917 to judge 0.3500, a 3.8x rise), and the relative ranking shifts further in `ours_utility`'s favor — it edges out both `lru` (0.3417) and the `no_forget` ceiling itself (0.3333). We do not present this as a confirmed stronger win: it is a raw mean on the same n=120 sample and would need the same bootstrap treatment as Section 6.6 before being cited as decisive.

### 6.9 Qualitative traces

Tracing the same gold-evidence memory across policies surfaced 8 cases where eviction under `ours` directly caused a wrong answer that `ours_utility` (having kept the memory) answered correctly. One representative example: for the question "When did Jon lose his job as a banker?" (reference: "19 January, 2023"), the evidence memory "Jon lost his job as a banker the day before the conversation" was evicted under `ours`, and the model answered "I don't have that information yet" — not a bad guess, direct proof the memory was genuinely gone. With the same memory retained under `ours_utility`, the model answered "2023-01-19" correctly. We also flag, rather than quietly omit, one EM-vs-judge disagreement that looks like a plausible *judge* error (reference "three years" judged equivalent to prediction "2019" — only correct if the judge silently infers a conversation date it was never shown), since the judge prompt does not see memory timestamps.

### 6.10 Does the diagnosis hold on LongMemEval?

We initially assumed LongMemEval lacked a usable evidence-linkage field and excluded it from the diagnostic in Sections 6.3–6.5. On direct inspection this assumption was wrong: the evidence field matches a haystack-session identifier for 100% of LongMemEval memories. We correct this and extend the diagnostic:

| Policy | Evidence retention rate (LongMemEval, N=92 covered questions) |
|---|---|
| no_forget / fifo / lru | 1.0000 |
| ours | 0.9783 |
| ours_utility | 0.9891 |

`no_forget`, `fifo`, and `lru` all achieve *perfect* retention here, precisely explaining why LongMemEval showed little downstream QA effect in Section 6.2: its small conversations rarely fill the storage budget, so naive policies almost never need to evict evidence in the first place. `ours` still measurably underperforms via the same TTL-only mechanism seen on LoCoMo (zero Action-head evictions), and `ours_utility` still helps — the same causal story, present but of smaller magnitude on a benchmark where the failure mode has less room to manifest.

### 6.11 Does eviction cause hallucination, or just wrong answers?

Every result in Sections 6.2–6.10 scores only LoCoMo's roughly 1,540 answerable QA pairs. LoCoMo also ships approximately 446 category-5 "adversarial" pairs — genuinely unanswerable questions paired with a plausible-sounding wrong decoy answer — which we excluded entirely from the main comparison (Section 6.1) rather than using to ask a different question: when eviction removes real evidence, does the system hallucinate against the decoy, or does it honestly refuse? We measure this with the refusal precision/recall/F1 methodology REMem (Shu et al., 2026) uses for exactly this purpose: on a balanced 120 answerable plus 120 adversarial question sample per policy, we classify every answer as a refusal (a deterministic phrase match, matching the exact behavior our own QA prompt instructs) and score against the true unanswerable label.

| Policy | Precision | Recall | F1 |
|---|---|---|---|
| no_forget | 0.718 | 0.892 | 0.796 |
| fifo | 0.639 | 0.883 | 0.741 |
| lru | 0.709 | 0.892 | 0.790 |
| ours | 0.648 | 0.892 | 0.751 |
| **ours_utility** | **0.726** | 0.883 | **0.797** |

Recall (does the system refuse a genuinely unanswerable question at all) is essentially flat across policies, 0.883–0.892 — confirmed not to be a coincidence by bootstrap significance (n=240, 10,000 resamples): recall differences are not the story here. Precision (when the system does refuse, was the question actually unanswerable) is different: `ours` and `fifo` false-refuse genuinely answerable questions significantly more often (63.9–64.8%) than `no_forget`/`lru`/`ours_utility` (70.9–72.6%) — `ours_utility` beats both `ours` (p<0.001) and `fifo` (p<0.001) on this metric, and `no_forget` beats both too (p=0.001 and p<0.001).

This is an independent confirmation of the Sections 6.3–6.5 root cause, via a completely different metric family that makes no reference to EM, F1, or any evidence-linkage field: when `ours`/`fifo` evict a question's real supporting evidence, the model does not only get the EM/F1 score wrong — it correctly, from its own epistemic standpoint, says it does not have that information, which this metric counts as a false refusal precisely because the eviction, not the model's reasoning, is what made an otherwise-answerable question unanswerable. The fix that already worked for EM/F1 again lands closest to the ceiling here.

### 6.12 Error-mode taxonomy

Every quantitative result above is a rate; the qualitative examples in Section 6.9 are hand-picked, not a systematic sample. Following REMem's error analysis (100 sampled errors, bucketed into named categories), we drew a reproducible 100-question sample of wrong (judge-negative) predictions pooled across all 5 policies and both benchmarks and categorized each one.

| Category | % | Meaning |
|---|---|---|
| REFUSAL | 54% | False refusal — "I don't have that information" on an answerable question |
| WRONG_VALUE | 21% | Confident but incorrect specific fact or hallucinated detail |
| INCOMPLETE | 15% | Captured only part of a multi-item reference answer |
| DATE_OFF_BY_ONE | 5% | Date/relative-date conversion off by ~1 day/unit |
| REASONING_ERROR | 2% | Dropped a comparative/causal relationship in the question |
| JUDGE_ERROR_CANDIDATE | 3% | Substantively correct; likely a judge scoring mistake |

False refusal is the dominant failure mode by a wide margin — not wrong guesses, not date-arithmetic mistakes, not dropped multi-step reasoning. This quantifies what the Section 6.11 refusal-precision result already implied: most of what looks like "the model got the question wrong" is, on inspection, "the eviction policy removed the evidence and the model correctly reported that." We note the methodology limitation directly rather than presenting this as equivalent to REMem's human error analysis: this categorization is a single-pass manual read-through, not independently verified by a second rater — there is no inter-rater agreement figure, and category boundaries on borderline partial answers are a considered judgment call, not a ground-truth label.

### 6.13 Does Fix #2 help every question type equally?

Every downstream EM/F1 number reported so far is one aggregate per policy. We sliced the same matched-sample predictions by LoCoMo's own category field (Multi-Hop, Single-Hop, Temporal, Open-Domain — verified against the raw benchmark file and cross-checked against REMem's own published per-category counts, an exact match) and LongMemEval's own question-type field.

The effect is concentrated, not uniform: `ours_utility` clearly beats `fifo`/`ours` on LoCoMo's Single-Hop questions (LLM-judge 0.537 vs. 0.204/0.259, N=54) and Multi-Hop questions (EM 0.083 vs. 0.021/0.021, matching the `no_forget` ceiling, N=48) — but on Temporal questions (N=16), EM is identical across all 5 policies (0.1875 in every case), i.e. this sample shows zero policy effect there. We note the sample-size limitation directly rather than over-interpreting: Open-Domain (N=2) and most LongMemEval categories (N=1–8) are too small in this pilot to support a per-category conclusion on their own. The qualitative takeaway survives the small-N caveat regardless: whatever mechanism drives Fix #2's improvement is not hitting every question type equally, and the aggregate numbers elsewhere in this paper should not be read as implying a uniform effect.

### 6.14 Is `no_forget` actually the ceiling?

Every claim in this paper that a policy "reaches the ceiling" means it matches `no_forget` — but `no_forget` retains every memory and still goes through ordinary top-5 retrieval, conflating two different costs: how much eviction costs, and how much limiting the prompt to the top-5 retrieved memories costs, independent of any eviction at all. Following REMem's own practice of anchoring results between an Oracle reference (given gold evidence only) and a Full-Context reference (the entire corpus, uncapped retrieval), we add both: `oracle` builds the memory store from only a QA pair's own gold-evidence memories (no retrieval noise, the true theoretical ceiling); `full_context` builds the store from every memory in the conversation, the same as `no_forget`, but with retrieval width set to the full store size so nothing is dropped by the usual top-5 cap.

| Benchmark | Policy | N | Mean EM | Mean F1 |
|---|---|---|---|---|
| LoCoMo | oracle | 107 | 0.1589 | 0.3473 |
| LoCoMo | full_context | 30 | 0.1333 | 0.2980 |
| LoCoMo | no_forget (for reference) | 120 | 0.0917 | 0.1957 |
| LongMemEval | oracle | 22 | 0.1364 | 0.1814 |
| LongMemEval | full_context | 25 | 0.0800 | 0.1525 |
| LongMemEval | no_forget (for reference) | 25 | 0.0800 | 0.1525 |

**`no_forget` is not the true ceiling on LoCoMo.** Bootstrap significance, matched on the same 107 questions where an oracle answer could be built: `oracle` significantly beats `no_forget` (EM +0.0654, 95% CI [+0.0187, +0.1215], p=0.0073; F1 +0.1416, 95% CI [+0.0829, +0.2033], p<0.0001). A policy that forgets nothing still leaves real accuracy on the table relative to what perfect retrieval could achieve — ordinary top-5 retrieval over a large, unevicted store is a genuine, independent bottleneck that no eviction policy examined in this paper, including `ours_utility`, was ever positioned to close, because eviction policy and retrieval quality are orthogonal problems. On LongMemEval, the same gap is not significant (p=0.36 on both metrics, N=22) — consistent with the Section 6.10 explanation that small stores leave little room for retrieval noise to matter there. `full_context` lands between `oracle` and `no_forget` on LoCoMo (N=22 matched on all three policies: EM 0.136 vs. oracle's 0.227 and no_forget's 0.182; F1 0.350 vs. 0.456 and 0.297), suggesting removing the top-5 cap recovers some but not all of the gap — but `full_context` vs. `no_forget` is the only pairwise comparison in this specific three-way subset that clears significance (F1 p=0.041; the other two pairs do not, p=0.061 and p=0.38, smaller than the N=107 headline comparison above because this restricts to the much smaller three-way-matched set), so we report it as suggestive, not confirmed.

This result does not change any of the Sections 6.4–6.7 conclusions about which eviction policy is best — it changes what "best" is measured against. `ours_utility` still significantly beats `fifo` and the original `ours`, and still matches `no_forget` among the policies actually compared. But the honest framing is: this paper closes most of the gap between a naive or miscalibrated eviction policy and the best eviction policy achievable at a matched storage budget; it does not close the separate, larger gap between any eviction policy and a system with genuinely better retrieval.

### 6.15 A real memory-system baseline: Mem0

Every comparison above pits our own eviction policies against each other. Following the practice REMem establishes (Section 2) of evaluating against independently implemented competing systems, we integrate Mem0 (Chhikara et al., 2025) genuinely: indexed all 10 LoCoMo conversations (5,882 dialogue turns) through Mem0's own extraction pipeline and answered the same 120-question sample used throughout Section 6.

Getting there required a real detour, disclosed in full rather than hidden. A cost calibration measured Mem0's own indexing calls — one LLM call per conversation turn, not per question — at $0.01063 per turn on GPT-4o via OpenRouter (60 real turns, measured from actual before/after account balance): $62.55 extrapolated to index all 5,882 turns, far beyond the available budget for this project. Mem0's own built-in provider mechanism was pointed instead at a hand-rolled local OpenAI-compatible server serving the same local Qwen2.5-7B-Instruct model already used as a baseline in Section 5.1 — zero marginal cost. Indexing throughput on the local model varied substantially by shared-GPU-node load, from approximately 4.4 seconds per turn in steady state (one conversation, 369 turns, 27.3 minutes) to over 30 seconds per turn during a period of severe node contention late in the run; total wall-clock across all 10 conversations was on the order of several hours.

This substitution introduces two real, measured limitations, not speculative ones. First, Mem0's open-source `add()` method has no working timestamp parameter (its own installed package documents this as a platform-only feature not supported in the open-source release); dates were injected as text prefixes instead, and the local model was directly observed getting this wrong at least once, resolving "yesterday" to 2026 instead of the conversation's actual 2023 — defaulting to real-world "today" rather than the stated context date. Second, the local model produced malformed JSON on 195 of 5,882 indexing calls (3.3%), each one silently contributing zero extracted memories for that turn (Mem0's own extraction code degrades gracefully on a parse failure rather than crashing, but the information is still lost). Both are disclosed confounds of the budget-driven substitution, not properties of Mem0 itself as normally deployed with a frontier extraction model.

| Policy | N | Mean EM | Mean F1 | Mean BLEU-1 |
|---|---|---|---|---|
| **mem0** | 120 | 0.0833 | **0.2275** | **0.1808** |
| no_forget | 120 | 0.0917 | 0.1957 | 0.1562 |
| ours_utility | 120 | 0.0917 | 0.1949 | 0.1549 |
| lru | 120 | 0.0833 | 0.1998 | 0.1586 |
| ours | 120 | 0.0500 | 0.1575 | 0.1116 |
| fifo | 120 | 0.0417 | 0.1294 | 0.0915 |

On raw F1/BLEU-1, mem0 beats every one of this project's own policies, including the `no_forget` ceiling. Bootstrap significance gives the more honest reading: mem0 is *not* significantly different from `no_forget`, `ours_utility`, or `lru` (all p>0.13) — statistically tied with our best result and the ceiling itself, not proven superior. It significantly beats `fifo` (F1 p<0.001) and `ours` (F1 p=0.005). Stated plainly: a real, independent memory system, even handicapped by a weaker local extraction model with a measured 3.3% data-loss rate and no working timestamp support, performs statistically comparably to this project's best eviction policy — not worse. We report this as measured, not reframed to favor this paper's own contribution.

## 7. Limitations

We state these directly rather than deferring them to an appendix.

- **The Mem0 comparison (Section 6.15) uses a weaker LLM for Mem0's own extraction than Mem0 is normally evaluated with** (a local Qwen2.5-7B in place of GPT-4.1-mini/GPT-4o, forced by budget, not chosen for a fairness advantage), with a measured 3.3% indexing data-loss rate from malformed JSON and no working timestamp support in Mem0's open-source release. Mem0 still performed statistically comparably to our best policy under this handicap; a like-for-like comparison with Mem0's typical setup could only make its relative showing stronger, not weaker, and is left for future work.
- **Retrieval quality, not just eviction policy, is a real bottleneck we do not fix** (Section 6.14): an oracle given only the correct evidence significantly outperforms `no_forget` on LoCoMo (p<0.01), meaning even a policy that forgets nothing is capped below what is achievable by ordinary top-5 retrieval over a large store. This paper's contribution is entirely about *what* to evict, not how retrieval itself ranks and selects from whatever survives eviction — a genuinely separate, unaddressed problem.
- **The censoring convention for real-conversation records** (Section 3.3, case 3) is a judgment call with no ground-truth alternative in the source data — a sensitivity analysis under a different censoring-time choice is a natural follow-up, not yet done.
- **The `ours_utility` vs. `lru` comparison is not statistically significant** at n=120 real LLM-scored questions (Section 6.6), even though the free-diagnostic proxy and the mechanistic AUC analysis both point the same direction.
- **The Importance head is a heuristic, not learned** — no ground-truth importance label exists in the dataset schema.
- **The residual gap between `ours` and `lru`/`fifo` at every TTL quantile** (Section 6.4) has a working hypothesis (LoCoMo's QA evidence may itself be recency-skewed, mechanically favoring recency-based selection) that is not yet verified.
- **The original dialogue-to-candidate-memory extraction pipeline** that produced the raw dataset is not included in this release; MemoryLifeBench ships the already-extracted records, not the extraction code.
- **No human validation** of extracted labels has been performed; all labels are programmatically derived.
- **LLM-judge scoring** (Section 6.8) is itself imperfect — see the flagged judge-error example in Section 6.9 — and its results are not yet subjected to the same significance testing as the EM/F1 headline claims.

## 8. Reproducibility

Headline results use 5 seeds (survival model) or 3 seeds (joint model); all reported means include standard deviation. 51 unit tests (31 added during the downstream-evaluation phase of this work) cover the pure-logic components behind every downstream claim in Section 6 — bootstrap significance arithmetic, the TTL-quantile cutoff behavior, all six eviction policies, and the evidence-retention report's table arithmetic. Code is released under the MIT license. Data is released under CC BY-NC 4.0 (non-commercial, with attribution to LoCoMo and LongMemEval required).

## 9. Conclusion

Framing memory-lifetime prediction as survival analysis, rather than a heuristic importance score, gives a small, deterministic, zero-token-cost model that ranks memory lifetimes substantially better than prompting frontier LLMs. But we show this ranking quality does not automatically translate into a good deployed forgetting policy: a real downstream QA evaluation caught a failure our own validated metric (C-index) could not have caught, because C-index is rank-only and never validates an absolute decision threshold. The fix that worked was not a better model, but a better *use* of a model we already had — ranked selection instead of independent thresholding, using a signal (Future-Utility) that was already trained but, before this work, was never consulted by the forgetting policy itself. We see this as the paper's most transferable finding for anyone building a memory system on top of a lifetime/utility model: validate the actual deployed decision rule against a task-level metric, not only the ranking metric the model was trained against. Two further checks kept this conclusion honest rather than self-serving: an Oracle comparison showed the eviction-policy fixes close the gap between policies, not the larger and separate gap to what better retrieval could achieve; and a real comparison against an independent memory system (Mem0) found our best policy statistically tied with it, not superior.

## References

Chhikara, P., Khant, D., Aryan, S., Singh, T., Yadav, D. (2025). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.

Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, Series B, 34(2), 187–220.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Larson, J. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.

Gutierrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.

Gutierrez, B. J., Shu, Y., Qi, W., Zhou, S., Su, Y. (2025). From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. arXiv:2502.14802.

Kvamme, H., Borgan, O., Scheel, I. (2019). Time-to-Event Prediction with Neural Networks and Cox Regression. *Journal of Machine Learning Research*, 20(129), 1–30.

Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., Fang, Y. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents. arXiv:2402.17753.

Packer, C., Fang, V., Patil, S. G., Lin, K., Wooders, S., Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.

Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior.

Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., Chalef, D. (2025). Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956.

Shu, Y., Jonnalagedda, S. P., Gao, X., Jimenez Gutierrez, B., Qi, W., Das, K., Sun, H., Su, Y. (2026). REMem: Reasoning with Episodic Memory in Language Agents. *ICLR 2026*, arXiv:2602.13530.

Tan, Z., Yan, J., Hsu, I.-H., Han, R., Wang, Z., Le, L. T., Song, Y., Chen, Y., Palangi, H., Lee, G., Iyer, A., Chen, T., Liu, H., Lee, C.-Y., Pfister, T. (2025). In Prospect and Retrospect: Reflective Memory Management for Long-Term Personalized Dialogue Agents. arXiv:2503.08026.

Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., Yu, D. (2024). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. arXiv:2410.10813.

Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. (2025). A-MEM: Agentic Memory for LLM Agents. arXiv:2502.12110.

Zhong, W., Guo, L., Gao, Q., Ye, H., Wang, Y. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory. *AAAI 2024*.

## Appendix A: Formal Definitions

Sections 4 and 6 describe the survival model and the eviction policies in prose above. This appendix gives explicit mathematical notation for both, matching the actual implementation precisely.

### A.1 Survival model

For memory record *i*, let *z_i* be its representation (a frozen BGE embedding for the single-task model of Section 4.1, or the fused embedding-plus-feature vector for the joint model of Section 4.2). The model learns a risk score *f*(*z_i*) (the MLP's scalar output) and defines a hazard function in the standard Cox proportional-hazards form:

h(t | z_i) = h_0(t) · exp(f(z_i))

S(t | z_i) = exp(−∫₀ᵗ h(u | z_i) du) = S_0(t) ^ exp(f(z_i))

where h_0(t) / S_0(t) are the baseline hazard/survival, estimated non-parametrically from the training set — f is the only part that is learned; the baseline is fit, not trained end-to-end.

**Observed data.** Each record has (T_i, δ_i): T_i is the duration from the statement's timestamp to the reference event, and δ_i ∈ {0, 1} indicates whether the event was actually observed (1) or right-censored (0), per the three cases in Section 3.3.

**Training objective.** The Cox partial log-likelihood over the uncensored subset D = {i : δ_i = 1}:

L(θ) = Σ_{i∈D} [ f(z_i) − log( Σ_{j∈R(T_i)} exp(f(z_j)) ) ]

where R(t) = {j : T_j ≥ t} is the risk set still "alive" at time t, fit via `pycox`'s standard Cox implementation rather than reimplemented from scratch.

**Evaluation metric (C-index).** Over comparable pairs E = {(i, j) : T_i < T_j, δ_i = 1}:

C = (1 / |E|) · Σ_{(i,j)∈E} 1[ f(z_i) > f(z_j) ]

the fraction of permissible pairs the model ranks in the correct order. C depends only on the *ordering* of f, not its scale — this is exactly why C-index cannot validate the absolute quantile cutoff below (Section 6.3, cause 1).

**Quantile TTL cutoff** (the Fix #1 target in Section 6.4):

TTL_q(z_i) = sup { t : S(t | z_i) ≥ q }

the latest time the record's own survival curve is still at or above survival probability q, capped at a maximum of 3,650 days for curves that never drop below q. q = 0.5 (the median) was the original hardcoded default; the fix makes q configurable.

### A.2 Eviction policies

Let O = {o_1, ..., o_n} be a conversation's memory objects, each with a predicted TTL, a utility probability in [0, 1] from the Future-Utility head, an action label from the Action head, a creation timestamp, and age(o, t) = t − created_at(o). Let t be the conversation's current timestamp.

**Threshold policy** (`ours`):

Active_ours(O, t) = { o ∈ O : action(o) ≠ forget AND age(o, t) ≤ predicted_ttl(o) }

an independent per-memory decision — no memory's inclusion depends on any other memory's score.

**Ranked top-N policies** (`fifo`, `lru`, `ours_utility`, `ours_combo`), all sharing one shape with capacity N = |Active_ours(O, t)| (matched to `ours`'s own natural budget, so every policy is compared at the same storage cost) and a policy-specific score:

Active_rank(O, N, score) = top-N of O ranked by score(o), descending

score_fifo(o) = created_at(o)

score_lru(o) = last_referenced(o) (−∞ if never referenced)

score_utility(o) = utility_prob(o) — the Fix #2 signal

score_combo(o) = 0.5 · utility_prob(o) + 0.5 · remaining_life_fraction(o, t)

remaining_life_fraction(o, t) = clip(1 − age(o, t) / predicted_ttl(o), 0, 1)

`remaining_life_fraction` is the continuous analogue of `ours`'s hard age-versus-TTL test — 1.0 for a brand-new memory, 0.0 once past its predicted TTL, linear in between.

## Appendix B: Use of Large Language Models

Two uses are disclosed separately below, since they differ in kind, not just degree.

**As infrastructure inside the method and evaluation, not as a writing aid.** GPT-4o, Gemini 2.5 Pro, and a locally hosted Qwen2.5-7B are evaluated as baselines (Section 5.1) — their outputs are an experimental subject, since part of this paper's claim is that our approach ranks memory lifetime better than prompting them. Separately, GPT-4o is used as a real component inside our own pipeline: the grounded-QA answering step that every Section 6 downstream number is scored against, LLM-judge rescoring (Section 6.8), and generating the synthetic source dialogues in the dataset itself (Section 3.2). All three are disclosed inline at first use throughout the paper; none of them constitutes the LLM helping to write this paper — they are the system and the evaluation harness being studied.

**In the research and writing process itself.** This project's code, experiments, and this manuscript were produced through extensive, direct collaboration with an AI coding assistant (Claude, Anthropic) across the full course of the work, not limited to grammar-checking or minor polishing. The assistant wrote the implementation code, ran and analyzed the experiments described throughout this paper, and drafted substantial portions of this text under the author's direction and review. This included catching and correcting the project's own prior mistakes rather than only producing new results — for example, an earlier, untested claim that LongMemEval had no evidence-linkage field (Section 6.10) and an earlier overstated claim that `ours_utility` beats `lru` outright (Section 6.6) were both identified and corrected during this same collaborative process. We disclose this directly because the actual degree of AI involvement here is substantial, and a reader assessing this work's provenance should know that plainly.

## Appendix C: Prompts

For reproducibility, every prompt used in this work is reproduced verbatim below, rather than only referenced by file path.

### C.1 TTL-prediction baseline prompt (Section 5.1)

Used identically for all three LLM-prompted-TTL baselines (GPT-4o, Gemini 2.5 Pro, local Qwen2.5-7B) — the same prompt on purpose, so "which model" is the only variable being compared.

> You are estimating how long a piece of memory stays useful for a personal AI assistant.
>
> Memory statement: "{text}"
>
> Question: starting from when this was said, how many days from now is this fact likely to remain true and useful to remember, before it becomes outdated or irrelevant? Some facts are permanent (use a large number like 3650 for "essentially forever"), some last weeks or months, and some are only relevant for a day or two.
>
> Answer with ONLY a single integer number of days. No words, no explanation.

### C.2 Grounded-QA prompt (Section 6, all downstream policy comparisons)

The only prompt that answers a question against retrieved memories — every EM/F1/BLEU-1/judge/refusal number in Section 6 traces back to a call using this template.

> You are a personal AI assistant answering a question using only memories retrieved from what the user has told you before. Each memory is dated. For "when" questions, do the arithmetic: if a memory dated 2023-05-08 says something happened "yesterday", the answer is 2023-05-07, not "recently" or "yesterday" — always convert relative time words (yesterday, last year, last week, etc.) into an absolute date/year using the memory's date, and give that absolute answer.
>
> Retrieved memories (most relevant first, each dated, each with how confident the system is that it's still true/relevant):
> {retrieved_memories_block}
>
> User's question: "{query}"
>
> Answer using ONLY the retrieved memories above. If none of them actually answer the question, say you don't have that information yet — do not guess or use outside knowledge. Give the SHORTEST correct answer: a bare date, name, number, or short phrase if that's all the question asks for — not a full sentence restating the question. Only use a full sentence if the question genuinely requires one.

Note the explicit refusal instruction ("say you don't have that information yet") — this is the exact phrase the refusal classifier in Section 6.11 was calibrated to detect.

### C.3 LLM-judge prompt (Section 6.8)

> You are grading whether a candidate answer to a question is substantively correct, given a known reference answer. Judge by MEANING, not exact wording — paraphrases, different units/formats for the same fact, and answers that include the correct fact plus extra correct context all count as correct. A vague, wrong, contradictory, or "I don't know" answer when the reference has a real answer counts as incorrect.
>
> Question: "{question}"
> Reference answer: "{reference}"
> Candidate answer: "{prediction}"
>
> Respond with exactly one word: CORRECT or INCORRECT. No explanation.

### C.4 Memory-extraction prompt (documented, not exercised by this paper's results)

Exists for a future live-deployment path (raw dialogue turns to extracted memory statements) — MemoryLifeBench's own records are already extracted, pre-labeled statements (Section 3), so this prompt is not called anywhere in producing any number reported in this paper. Included for completeness, not because it was used.

> You extract durable, memory-worthy statements from a single conversational turn, for a personal AI assistant's long-term memory store.
>
> Conversation turn (speaker: {speaker}): "{turn_text}"
>
> If this turn states a fact, preference, plan, or event worth remembering for future conversations, output it as a short, self-contained third-person statement (e.g. "User lives in Denver.", "User has a flight on March 3rd."). If it's small talk, a question with no lasting content, or nothing worth remembering, output exactly: NONE
>
> Output ONLY the extracted statement (or NONE). No explanation.
