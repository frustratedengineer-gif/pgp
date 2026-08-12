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
