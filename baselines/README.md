# Baselines

## Implemented (Week 3 -- survival/TTL prediction, compared by C-index)

| File | What it is | Trained? |
|---|---|---|
| `llm_prompted_ttl.py` | Prompts a local instruct model (Qwen2.5-7B-Instruct via ollama) for a predicted lifetime in days | No (zero-shot) |
| `chatgpt_prompted_ttl.py` | Same prompt, real GPT-4o via OpenRouter (default `openai/gpt-4o`) | No (zero-shot) |
| `gemini_prompted_ttl.py` | Same prompt, real Gemini via OpenRouter (default `google/gemini-2.5-pro`, check the file for the current slug) | No (zero-shot) |
| `bucket_classifier.py` | Logistic regression on BGE embeddings, predicts day/week/permanent | Yes, on observed-event subset of train |
| `heuristic_ttl.py` | Regex/keyword rules on raw text, no model | No |

`chatgpt_prompted_ttl.py` / `gemini_prompted_ttl.py` exist specifically to
answer the real reviewer question -- not "did you beat a local 7B model"
but "did you beat GPT-4o / Gemini, the thing I'd actually type into a chat
box myself". Same exact prompt as the local baseline, so the comparison
isolates the model, not the prompt. Both go through OpenRouter
(`baselines/_openrouter_client.py`) -- one API key, one OpenAI-compatible
endpoint, model chosen by slug (`openai/gpt-4o`, `google/gemini-2.5-pro`).
Requires `OPENROUTER_API_KEY` in the environment -- see `.env.example`.
`scripts/evaluate.py` includes them automatically once their score files
exist and skips them gracefully otherwise.

Run with `scripts/run_baseline.py --method <name>`; each writes
`artifacts/scores/<name>_<split>.json` in the score-direction convention
documented in `src/memorylife/evaluation/survival_metrics.py` (higher score
== predicted to survive longer), plus `<name>_token_usage.json` (prompt/
completion/total tokens per split -- a cost/efficiency metric reported
separately from accuracy).

Results: `results/tables/week3_results_table.md`,
`results/tables/llm_token_usage.md`.

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
