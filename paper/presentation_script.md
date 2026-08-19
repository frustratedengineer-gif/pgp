# MemoryLifeBench — Talk Script

**Format:** Academic talk / thesis-defense style, 15–20 minutes + Q&A
**Author:** Bhargav Shendge

---

## How this script is built (presentation-design notes)

Before writing the talk, five principles from how strong research talks are actually structured (conference-talk and thesis-defense norms) were applied on purpose:

1. **Tell a story, don't recite the paper's table of contents.** A section-by-section walkthrough ("Section 1... Section 2...") is the single most common way academic talks lose an audience. This paper already has a real narrative arc — build a good ranker → validate it → deploy it → **it fails** → diagnose why → fix it → check the fix honestly → compare to a real competitor. The talk is structured around that arc, not around the paper's table of contents.
2. **One idea per slide, one number per claim.** Every slide below has a single sentence it exists to prove. Supporting numbers are trimmed to the 2–3 that carry the point; the full tables stay in the paper and in backup slides.
3. **Front-load the result, don't save it for the end.** The headline finding (learned ranking ≠ good deployed policy, and the fix is structural, not a bigger model) is previewed in the first two minutes, not just at the conclusion. Audiences retain more when they know what they're being walked toward.
4. **Own the negative result on purpose.** The strongest part of this paper is that it reports a result that initially contradicts its own headline claim, then diagnoses and fixes it in public. A talk that hides this to look cleaner is a weaker talk — reviewers and committees respond well to visible intellectual honesty, and it's this paper's actual selling point.
5. **Defend, don't just present.** Because this is a defense-style talk, likely follow-up questions are anticipated with dedicated **backup slides** (not crammed into the main deck) — formal notation, the full six-policy definitions, and the extra significance tables. Main slides stay uncluttered; backups are ready the moment a question calls for them.

Timing below is paced for a **17-minute core talk**, leaving 3–5 minutes of the 15–20 minute window free for early interruptions or a slower pace, with backup slides ready for Q&A.

---

## SLIDE 1 — Title
**[00:00 – 00:30]**

**On screen:**
> MemoryLifeBench
> Memory Lifetime Prediction as a Time-to-Event Problem
> Bhargav Shendge

**Say:**
"Thank you. This talk is about a simple question that turns out to be surprisingly hard to answer well: when an AI assistant remembers something about you, how does it decide when that memory is no longer worth keeping? I'll show you a model that answers this question very well by one measure — and then show you that answering it well was not enough, walk through why, and show what actually fixed it."

---

## SLIDE 2 — The problem
**[00:30 – 01:30]**

**On screen:**
- Long-running AI assistants accumulate memories across sessions
- Storage isn't free, and unbounded context windows don't solve it
- Retrieval quality degrades as the store grows
- Stale or contradicted facts actively hurt answers if never removed

**Say:**
"Personal AI assistants that run across many sessions keep extracting facts about you — where you live, what you're working on, what you said last week. That store keeps growing. You can't just keep everything forever: retrieval gets noisier as the store grows, and stale facts don't just sit there harmlessly, they actively produce wrong answers when they contradict something newer. So at some point, the system has to decide: what gets forgotten, and when?"

---

## SLIDE 3 — How existing systems answer this
**[01:30 – 02:15]**

**On screen:**
- Generative Agents: hand-tuned recency/importance/relevance score, decayed
- MemGPT: OS-style paging between working set and external storage
- Mem0: heuristic consolidation, frequently discards by internal decision
- **None of them predict *when* a memory will stop being useful**

**Say:**
"Existing memory systems all answer this with some flavor of a scalar importance score — a recency-weighted heuristic, a paging scheme, a consolidation rule. What none of them do is predict an actual **time**. They tell you a memory is 'less important now,' not 'this fact will likely stop mattering in about three weeks.' That's the gap this work targets."

---

## SLIDE 4 — Our reframing
**[02:15 – 03:15]**

**On screen:**
- Treat every memory as a **survival-analysis subject**
- Born when stated → dies when invalidated, contradicted, or never referenced again
- Right-censored if we never observe the "death"
- Inherits a mature toolkit: Cox proportional hazards, concordance index, calibration vs. discrimination

**Say:**
"We reframe this as survival analysis — the same statistical framework used for patient survival times and equipment failure times. Every memory is 'born' when it's stated, and 'dies' when it becomes invalid or is never referenced again. A lot of the time we don't observe the death within our data window — that's censoring, and it's handled correctly rather than ignored. This gets us a mature, well-understood toolkit for free: proportional hazards models, concordance index, calibration diagnostics. And — this becomes important later in the talk — that toolkit draws a sharp, textbook distinction between a model that **ranks** well and a model that's **calibrated** on an absolute cutoff. Hold onto that distinction; it's the hinge the whole second half of this talk turns on."

---

## SLIDE 5 — The dataset: MemoryLifeBench
**[03:15 – 04:00]**

**On screen:**
| Source | Train | Val | Test |
|---|---|---|---|
| Synthetic (provable lifetimes) | 3,199 | 256 | 265 |
| LongMemEval | 3,162 | 359 | 375 |
| LoCoMo | 1,936 | 324 | 276 |
| **Total** | **8,297** | **939** | **916** |

- Split **by conversation** (568/71/71), verified no leakage

**Say:**
"To train and evaluate this, we built MemoryLifeBench: just over 10,000 memory records with time-to-event labels, combining synthetic dialogues where we know the true lifetime by construction, with real conversations pulled from two published benchmarks, LoCoMo and LongMemEval. Everything is split by conversation, not by record, and we verified there's no leakage — no conversation's memories appear in more than one split."

---

## SLIDE 6 — The model
**[04:00 – 05:00]**

**On screen:**
- Frozen sentence embedding (BGE-base, 768-d) → small MLP → log hazard
- **213,889 parameters** — no fine-tuning, no generation
- Trained with Cox partial-likelihood loss, handles right-censoring correctly
- Zero token cost by construction

**Say:**
"The core model is deliberately small: a frozen sentence embedding feeds a 214-thousand-parameter MLP that outputs a single hazard score, trained with the standard Cox partial-likelihood objective. No fine-tuning of the embedding, no generation, nothing autoregressive — which matters, because it means this model costs zero tokens to run, versus every LLM-prompted baseline we compare against."

---

## SLIDE 7 — Extending to a joint model
**[05:00 – 05:45]**

**On screen:**
- Six frozen auxiliary extractors (intent, entities, temporal, emotion, novelty, contradiction) fused with the embedding
- Three heads sharing one representation: **Lifetime · Action (store/update/merge/forget) · Future-Utility**
- Still under 500K trainable parameters total

**Say:**
"We also built a joint version: six off-the-shelf feature extractors are fused with the embedding, feeding three heads that share one representation — the same Lifetime head as before, an Action head that classifies store/update/merge/forget, and a Future-Utility head predicting whether a memory gets retrieved again. All three trained jointly, still under half a million parameters total. The Future-Utility head becomes the hero of this talk's second act — remember that name."

---

## SLIDE 8 — Headline ranking result
**[05:45 – 07:15]**

**On screen:**
| Method | C-index (test) |
|---|---|
| **Our survival model** | **0.7218** |
| Day/week/permanent classifier | 0.6298 |
| GPT-4o (prompted TTL) | 0.5411 |
| Qwen2.5-7B (prompted TTL) | 0.5207 |
| Gemini 2.5 Pro (prompted TTL) | 0.4806 |
| Recency-frequency heuristic | 0.4753 |

- Beats every baseline, **p < 0.001**, all pairwise bootstrap tests
- vs. GPT-4o: +0.18 C-index, 95% CI [+0.13, +0.22]

**Say:**
"Here's the headline result from the first half of the paper. Our model beats every baseline — including prompting GPT-4o and Gemini 2.5 Pro directly for a lifetime estimate — by a wide, statistically significant margin. Every single comparison clears p less than 0.001 under bootstrap testing. Against GPT-4o specifically, that's an 18-point C-index gap. And it costs zero tokens to run, versus real, metered API spend for the LLM baselines."

---

## SLIDE 9 — Beyond the headline number
**[07:15 – 08:15]**

**On screen:**
- **Determinism:** our model has zero variance across paraphrases; GPT-4o's coefficient of variation is 0.38 (44% of records CV>0.5); Qwen's is 0.75 (85%)
- **5-seed stability:** 0.7312 ± 0.0131 — not a lucky draw
- **Joint model:** 0.7553 ± 0.0045 — a real improvement, tighter spread too

**Say:**
"Three more things worth a beat each. First, determinism: reword the same fact three ways, and our model gives the same answer every time — the LLM baselines don't, sometimes wildly. That's a real liability if a forgetting policy changes its mind depending on how a fact happened to be phrased. Second, this isn't a lucky seed — five seeds land within a tight band. Third, the joint multi-task version improves further, to 0.7553, with an even tighter spread than the single-task model."

---

## SLIDE 10 — The harder question
**[08:15 – 09:15]**

**On screen:**
- A good C-index proves the model **ranks** memories well
- It does **not** prove the resulting **policy** — what actually gets kept or deleted — preserves downstream answer quality
- Next: match everyone to the same storage budget, and ask real questions against what survives

**Say:**
"So far, everything is validated on the model's own metric — how well it ranks. But a memory system doesn't rank memories, it deletes some of them. Does good ranking actually translate into a good deployed policy? To test this honestly, we matched every policy to the same final storage budget per conversation, and asked real LoCoMo and LongMemEval questions against whatever memories survived — scored by exact match and F1 by GPT-4o."

---

## SLIDE 11 — The surprise negative result
**[09:15 – 10:15]**

**On screen:**
| Policy | Mean EM | Mean F1 |
|---|---|---|
| no_forget (ceiling) | 0.1706 | 0.3007 |
| fifo | 0.1316 | 0.2347 |
| lru | 0.1245 | 0.2319 |
| **ours (our original policy)** | **0.1102** | **0.2054** |

**Say:**
"This is the moment the paper turns. Our original policy — the one built on the model that just won every ranking comparison — was the **worst** policy tested. Worse than first-in-first-out. Worse than least-recently-used. Well below the no-forgetting ceiling. That's the opposite of what the C-index results would have predicted, and we didn't smooth over it — we chased it down."

---

## SLIDE 12 — Root-cause diagnosis
**[10:15 – 11:30]**

**On screen:**
- Free diagnostic: does the gold-evidence memory literally survive eviction?
- `ours` retains only **66.9%** of needed evidence vs. lru's 76.8%, fifo's 72.5%
- **Cause 1 — miscalibrated threshold:** TTL cutoff = survival curve's *median* = a coin flip by construction (mean shortfall: 114 predicted days vs. 164 actually needed)
- **Cause 2 — wrong decision structure:** 100% of losses came from TTL expiry, 0% from the Action head — an independent per-memory threshold, not a ranked top-N like fifo/lru use

**Say:**
"We built a free diagnostic — no LLM calls — that just checks set membership: did the memory a question actually needs survive eviction? Our policy kept only two-thirds of what it needed. Two causes, and we verified both directly rather than guessing. One: the TTL cutoff we used was the survival curve's median — which is a coin-flip cutoff by definition, so on average it evicts about half the things it should keep. Two, and this is the deeper one: our policy makes an independent per-memory decision — each memory judged alone against its own threshold — while fifo and lru always keep their best N by a *ranked* score. A threshold can waste capacity on a mediocre memory that happens to land on the 'keep' side, while something genuinely important with slightly noisier scoring falls just short."

---

## SLIDE 13 — The fix, and why it works
**[11:30 – 12:45]**

**On screen:**
- Fix: rank by the already-trained **Future-Utility head**, keep top-N — same structure fifo/lru use, a trained score instead of a heuristic one
- Beats every baseline at every budget tested, including the original unfixed setting
- **Mechanistic proof:** utility_prob AUC = 0.67 predicting real evidence relevance; predicted-TTL AUC = **0.29** — actually backwards

**Say:**
"The fix wasn't a bigger model — it was a better use of a model we already had. We already had a Future-Utility head, trained to predict whether a memory gets used again, but it was only ever used to rerank retrieval — never consulted by eviction. We switched eviction to rank by that score and keep the top N, the same ranked structure fifo and lru already use. That alone beat every baseline at every storage budget we tested. And we didn't stop at 'it works' — we checked *why*: the utility signal has real predictive power for whether a memory is actual QA evidence, AUC 0.67. The old TTL-based signal scores 0.29 — below chance, actually pointing the wrong direction. That's a complete causal explanation, not a correlation we're hoping holds up."

---

## SLIDE 14 — Confirming it honestly
**[12:45 – 14:00]**

**On screen:**
| Comparison | EM diff | p |
|---|---|---|
| fixed policy vs. fifo | +0.050 | 0.002 |
| fixed policy vs. original policy | +0.042 | 0.006 |
| fixed policy vs. no-forget ceiling | +0.000 | 1.000 (tied) |
| **fixed policy vs. lru** | +0.008 | **0.367 — not significant** |

- Confirmed on a 4th metric (BLEU-1) and by refusal-precision/F1 (following REMem's methodology)
- Error taxonomy on 100 sampled failures: **54% are false refusals**, not wrong guesses

**Say:**
"On a real sample of 120 GPT-4o-scored questions, with proper paired bootstrap testing — not raw means — the fixed policy significantly beats fifo and the original policy, and is statistically indistinguishable from the no-forgetting ceiling. It is **not** significantly better than lru at this sample size, and we report that plainly rather than dressing it up — even though a larger free-diagnostic sweep and a mechanistic explanation both point the same direction. We also ran an error taxonomy on 100 sampled failures across every policy: 54% of what looks like 'the model got it wrong' is actually a false refusal — the system correctly saying it doesn't have information that eviction had already removed. Most of the story here is about what survives eviction, not about the model reasoning badly."

---

## SLIDE 15 — Two honesty checks
**[14:00 – 15:15]**

**On screen:**
**1. Is no-forget really the ceiling?**
- Oracle (only the correct evidence, no retrieval noise) significantly beats no-forget: +0.14 F1, p<0.0001
- Retrieval quality is a real, separate bottleneck this paper does not fix

**2. How do we compare to a real, independent memory system?**
- Mem0, integrated genuinely (real indexing, real extraction pipeline)
- Statistically **tied** with our best policy and the ceiling (p>0.13); significantly beats fifo and the original policy

**Say:**
"Two more checks, both designed to keep us honest rather than to make the result look better. First: is 'keep everything' really the best anything could do? No — an oracle given only the correct evidence, with no retrieval noise, significantly beats the no-forget ceiling. So ordinary top-5 retrieval over a large store is its own separate bottleneck, one this paper's eviction fixes were never positioned to close. Second: we integrated Mem0, a real independently-built memory system, end to end — not just discussed it. Our best policy is statistically tied with it, not superior. We report that as measured, not reframed to sound better than it is."

---

## SLIDE 16 — Limitations
**[15:15 – 16:00]**

**On screen:**
- Mem0 comparison used a weaker local model for its own extraction, forced by budget — a like-for-like setup could only help Mem0's showing
- No human validation of extracted labels yet
- `ours_utility` vs. `lru` gap not significant at current sample size
- Censoring convention for real conversations is a judgment call, not a given fact

**Say:**
"Being direct about what this doesn't yet show: the Mem0 comparison used a weaker local model for Mem0's own extraction, purely a budget constraint — a fair fight with its usual setup could only make Mem0 look stronger, not weaker. No human has yet validated the extracted labels. The comparison against lru specifically needs a bigger sample before it clears significance. And the censoring convention for real conversations is a judgment call we flag rather than present as settled."

---

## SLIDE 17 — Conclusion
**[16:00 – 17:00]**

**On screen:**
- Survival analysis gives a small, deterministic, zero-cost model that ranks memory lifetimes far better than prompting frontier LLMs
- **But ranking quality alone did not guarantee a good policy** — a downstream test caught what C-index structurally could not
- The fix was not a bigger model — it was validating the actual deployed decision, not just the metric it was trained on
- Two honesty checks — an Oracle ceiling and a real Mem0 comparison — kept the conclusion from overselling itself

**Say:**
"To close: framing this as survival analysis gives a tiny, deterministic, zero-token model that beats prompting frontier LLMs at ranking memory lifetimes, by a wide and significant margin. But the paper's real contribution is what happened next — ranking quality did not automatically produce a good deployed policy, we caught that with a downstream test the ranking metric structurally could not have caught, and the fix that worked was not a bigger model, it was validating the actual decision a deployed system makes, not just the metric it was trained against. If there's one transferable lesson for anyone building a memory system on top of a learned model, it's that one. Thank you — happy to take questions."

---

## BACKUP SLIDES (for Q&A)

### B1 — Formal notation
- Hazard: h(t|z) = h₀(t)·exp(f(z)); Survival: S(t|z) = S₀(t)^exp(f(z))
- Cox partial log-likelihood over uncensored set D, risk set R(t) = {j : T_j ≥ t}
- C-index over comparable pairs E = {(i,j): T_i<T_j, δ_i=1}: fraction correctly ordered
- Quantile TTL: TTL_q(z_i) = sup{t : S(t|z_i) ≥ q}, capped at 3,650 days

### B2 — All six eviction policies, formally
- `no_forget`: retain everything
- `fifo`: top-N by created_at
- `lru`: top-N by last_referenced (−∞ if never referenced)
- `ours` (original): action ≠ forget AND age ≤ predicted_ttl — independent per-memory threshold
- `ours_utility` (the fix): top-N by utility_prob
- `ours_combo`: top-N by 0.5·utility_prob + 0.5·remaining_life_fraction — underperforms pure utility; the utility signal alone does the work

### B3 — Quantile sweep in full
| Q | ours storage kept | ours retention | fifo | lru | ours_utility |
|---|---|---|---|---|---|
| 0.5 | 70.9% | 0.6687 | 0.7247 | 0.7676 | **0.8765** |
| 0.2 | 91.3% | 0.9080 | 0.9225 | 0.9517 | **0.9663** |
| 0.1 | 95.2% | 0.9502 | 0.9670 | 0.9709 | **0.9877** |
| 0.05 | 97.1% | 0.9716 | — | — | — |

### B4 — Question-type breakdown (matched sample)
- Single-Hop (N=54): fixed policy 0.537 judge score vs. 0.204 (fifo) / 0.259 (original) — clear win
- Multi-Hop (N=48): fixed policy 0.083 EM, matches the no-forget ceiling exactly
- Temporal (N=16): **0.1875 EM across all five policies — zero measured effect**, sample too small to call it a real null
- Open-Domain and most LongMemEval categories: N=1–8, too small for a per-category claim

### B5 — Mem0 integration cost detail
- Real calibration: 60 turns, $0.6381 spent → $0.01063/turn → $62.55 extrapolated for full GPT-4o indexing (budget-infeasible for a self-funded project)
- Substituted a free, self-hosted local Qwen2.5-7B for Mem0's own extraction calls only; QA-answering step stayed on real GPT-4o for a fair comparison basis
- Two disclosed confounds from that substitution: no working timestamp parameter in Mem0's open-source release (local model once resolved "yesterday" to 2026 instead of the true 2023) and 3.3% of indexing calls (195/5,882) produced malformed JSON, silently contributing zero memories for that turn

### B6 — LLM-judge cross-check
- 22.1% of predictions (160/725) marked wrong by exact-match were judged substantively correct
- Under the judge metric, scores rise 3.6–4.4x across every policy, and the fixed policy edges out both lru and the no-forget ceiling — reported as suggestive (same-sample raw means), not yet bootstrap-confirmed

### B7 — Anticipated tough questions
- *"Why not just use a bigger/frontier model for the Lifetime head instead of fixing the policy?"* → The mechanistic AUC result (Slide 13) shows the problem wasn't model capacity — the TTL signal was inversely correlated with real evidence relevance. A bigger model trained on the same miscalibrated absolute-threshold objective would not fix a structural decision-rule problem.
- *"Isn't 'tied with Mem0' a weak result to headline?"* → It's reported because it's true and because our comparison handicapped Mem0 (weaker local extraction model); a fair-setup comparison is explicitly named as future work, not hidden.
- *"How do you know the C-index gap over GPT-4o isn't just prompt sensitivity?"* → Same TTL-prediction prompt (Appendix C.1) used identically across all three LLM baselines, so "which model" is the only variable being compared.
