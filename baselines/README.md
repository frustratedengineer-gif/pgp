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

Responses are cached on disk under `artifacts/llm_cache/<method>/` (gitignored,
regenerable), keyed by `sha256(model + prompt)`. Cache hits don't count
against `llm_token_usage.md` -- Week-4 experiments (multi-seed reruns,
paraphrase consistency checks) re-query the same records repeatedly, and
re-billing OpenRouter for an identical prompt every time would be wasteful.

## Implemented (Week 6 -- downstream memory-system comparison)

Corrected here, since this section was stale through most of Week 6:
`no_forget.py`, `fifo.py`, and `lru.py` ARE implemented -- standalone
`sweep(store, audit_log, capacity, **kwargs) -> list[str]` functions
(matching the shape `fifo.py` above shows) that forget by capacity, using
the real memory store/audit log from `src/memorylife/memory/`. The actual
downstream QA-accuracy comparison harness
(`scripts/run_downstream_qa_eval.py`) reimplements the same three
policies' selection logic directly as pure set/ranking functions
(`final_active_ids`, see `docs/architecture.md` Appendix A.2 in
`paper/draft.md`) rather than calling these `sweep()` functions, since
the harness needs "what's the final active set" not "what got evicted in
this one sweep step" -- both exist, are consistent with each other, and
`final_active_ids` is the one every reported EM/F1/significance number
actually traces to (`results/tables/week6_downstream_qa*.md`).

`mem0_wrapper.py` is also implemented (Week 6, reviewer gap #1) -- a real
integration of Mem0 (Chhikara et al., 2025), configured to route its own
LLM extraction calls through either OpenRouter/GPT-4o or (the path
actually used for the full 10-conversation LoCoMo comparison, after a
real cost calibration showed the GPT-4o path would cost ~$62) a local
Qwen2.5-7B-Instruct server (`scripts/local_llm_server.py`). See the
module's own docstring for the disclosed limitations (no working
timestamp parameter in Mem0 OSS, a measured 3.3% JSON-parse failure rate
on the local model) and `README.md`'s "A real memory-system baseline:
Mem0" section / `paper/draft.md` Section 6.15 for the full results.

## Still not implemented

`generative_agents_importance.py`, `memgpt_wrapper.py`, `locomo.py`, and
`longmemeval.py` remain empty placeholders. The first two would be
further real memory-system baselines (Generative Agents' importance
scoring, MemGPT) alongside Mem0 -- not attempted this project given the
real cost/integration effort Mem0 alone took (see `scripts/
calibrate_mem0_cost.py` and MemGPT's own heavier agent-server
architecture, harder to stand up than Mem0's library-call interface).
`locomo.py`/`longmemeval.py` here would be baseline-style wrappers around
those benchmarks specifically -- distinct from (and not to be confused
with) `src/memorylife/data/build_benchmark.py`'s `load_locomo_qa`/
`load_longmemeval_qa`, which ARE implemented and are what
`scripts/run_downstream_qa_eval.py` actually uses.
