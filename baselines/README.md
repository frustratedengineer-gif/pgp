# Baselines

## Implemented (Week 3 -- survival/TTL prediction, compared by C-index)

| File | What it is | Trained? |
|---|---|---|
| `llm_prompted_ttl.py` | Prompts a local instruct model (Qwen2.5-7B-Instruct via ollama) for a predicted lifetime in days | No (zero-shot) |
| `bucket_classifier.py` | Logistic regression on BGE embeddings, predicts day/week/permanent | Yes, on observed-event subset of train |
| `heuristic_ttl.py` | Regex/keyword rules on raw text, no model | No |

Run with `scripts/run_baseline.py --method <name>`; each writes
`artifacts/scores/<name>_<split>.json` in the score-direction convention
documented in `src/memorylife/evaluation/survival_metrics.py` (higher score
== predicted to survive longer).

Results: `results/tables/week3_results_table.md`.

## Not yet implemented (Week 5 -- downstream memory-system comparison)

`no_forget.py`, `fifo.py`, `lru.py`, `generative_agents_importance.py`,
`mem0_wrapper.py`, `memgpt_wrapper.py`, `locomo.py`, `longmemeval.py` are
placeholders for the end-to-end QA-accuracy-vs-memory-size comparison
(architecture diagram's downstream retrieval/LLM box), not the Week-3
survival-prediction comparison. They compare storage/forgetting *policies*,
not lifetime *predictions*, and depend on the memory store
(`src/memorylife/memory/`) and retriever (`src/memorylife/retrieval/`)
which are also not built yet. Commit hashes / versions of any wrapped
third-party systems (mem0, MemGPT) will be recorded here once those files
have real content.
