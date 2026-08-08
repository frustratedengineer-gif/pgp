# Architecture

Figure: `figures/architecture.pdf`. Boxes are colour-coded in the original:
grey = existing components (off-the-shelf models), green = learned modules
(ours), orange = lifecycle & audit (ours).

| Box | What it is | Status | Code |
|---|---|---|---|
| Sentence Encoder (E5/BGE) | 768-d embedding of the input statement | **Built** | `src/memorylife/encoders/bge.py` |
| Semantic Feature Extractors (Intent, Entities, Temporal, Emotion/Preference, Novelty, Contradiction) | Auxiliary features fused with the embedding | Not built | `src/memorylife/features/*.py` (stubs) |
| Feature Fusion -> fused vector z | Combines embedding + features | Not built (currently z == raw embedding, no fusion) | `src/memorylife/fusion/*.py` (stubs) |
| Joint Lifecycle Predictor: **Lifetime head** (hazard h(t\|z) -> S(t\|z), TTL) | Survival model | **Built** (Week 3) | `src/memorylife/heads/survival.py`, `src/memorylife/losses/cox_partial.py` |
| Joint Lifecycle Predictor: Importance head | Score in [0,1] | Not built | `src/memorylife/heads/importance.py` (stub) |
| Joint Lifecycle Predictor: Future-utility head | P(retrieved in [t, t+delta]) | Not built | `src/memorylife/heads/future_utility.py` (stub) |
| Joint Lifecycle Predictor: Action head | store/update/merge/forget | Not built | `src/memorylife/heads/action.py` (stub) |
| Memory Object | `{text, embedding, importance, TTL, type, action, provenance}` | Not built | `src/memorylife/memory/memory_object.py` (stub) |
| Memory Store (vector DB + metadata), periodic reflection, self-compaction, forget+audit log | The lifecycle system | Not built | `src/memorylife/memory/*` (stubs) |
| Retriever (sim(q,e) + importance + utility) | Downstream retrieval | Not built | `src/memorylife/retrieval/*` (stubs) |
| LLM -> grounded answer | Downstream QA | Not built | `src/memorylife/inference/*` (stubs) |

## What Week 3 actually is, precisely

A single learned component: raw BGE embedding `e` -> small MLP -> log
partial-hazard, trained with the Cox partial-likelihood loss on
`(duration_days, event_observed)` pairs from MemoryLifeBench, correctly
handling right-censored records (`src/memorylife/data/censoring.py`).
Evaluated by concordance index against three baselines
(`baselines/`). No feature fusion, no other heads, no memory store, no
retrieval loop yet -- those are Weeks 4-5 per `docs/reproducibility.md`'s
known-gaps section.
