"""
Reviewer gap #1 (comparing against REMem, ICLR 2026, and against this
project's own README/reproducibility.md, which listed `mem0_wrapper.py`
as an unimplemented stub through Week 6): a real end-to-end memory-system
baseline, not just our own eviction policies compared against each other.

Mem0 (Chhikara et al., 2025) is configured here with:
  - LLM: local, by default -- Mem0's built-in "vllm" provider
    (`mem0.llms.vllm.VllmLLM`) is just an OpenAI-client pointed at a
    configurable `vllm_base_url` (verified by reading the installed
    package source, not assumed from docs), so it works against
    `scripts/local_llm_server.py`'s hand-rolled OpenAI-compatible server
    serving Qwen2.5-7B-Instruct locally -- zero marginal API cost. This
    exists because a real calibration run (see
    `scripts/calibrate_mem0_cost.py`) measured Mem0's OWN indexing calls
    (per conversation TURN, not per question) at ~$0.0106/turn on
    GPT-4o via OpenRouter -- ~$62 to index all 5,882 LoCoMo turns across
    10 conversations, far more than this project's remaining OpenRouter
    budget supports. `use_local_llm=False` switches back to the
    OpenRouter/GPT-4o path (mem0's `openai` provider auto-detects
    `OPENROUTER_API_KEY` from the environment and routes through
    `https://openrouter.ai/api/v1` automatically -- see
    `mem0/llms/openai.py`) for anyone re-running this with a bigger
    budget than a student stipend affords.
  - Embedder: local HuggingFace sentence-transformers (default
    `multi-qa-MiniLM-L6-cos-v1`), NOT OpenAI, since OpenRouter has no
    embeddings endpoint and this project has never held a direct OpenAI
    key -- a real, disclosed deviation from Mem0's out-of-the-box default
    (`text-embedding-3-small`), forced by infrastructure, not chosen for
    a fairness advantage.
  - Vector store: local Qdrant (on-disk, no server), Mem0's own default.

**A second real, disclosed limitation, on top of the local-LLM
substitution above**: using Qwen2.5-7B for Mem0's OWN extraction/update
reasoning (not just as this project's own answering model elsewhere) is
itself a deviation from how Mem0 is normally evaluated (REMem's own
Mem0 comparison uses GPT-4.1-mini). A weaker local extraction model could
plausibly make Mem0 look worse than it would with its "real" default
LLM -- this is a genuine confound of the budget-driven substitution, not
swept under the rug.

**A real, disclosed limitation of Mem0's open-source package, discovered
while integrating it, not assumed**: `Memory.add()`'s `timestamp`
parameter is documented in the installed package itself as "Platform-only
temporal parameter. Not supported in OSS." Mem0 OSS cannot be told a
message's real historical date via any API parameter. We work around
this the only way available -- prefixing each turn's text with its real
date (`[8 May 2023, 1:56 pm] Caroline: ...`), the same textual-date
convention this project's own baselines and REMem's gist extraction use
-- but Mem0's own extraction step may or may not act on that convention,
and it is never stored as structured, queryable metadata the way our own
`predicted_ttl_days`/timestamps are. This is a structural disadvantage
for Mem0 on temporal-reasoning questions specifically, not a bug in this
integration -- disclosed here so it isn't silently absorbed into a
"Mem0 is worse" headline number without the caveat attached.
"""
import os

from mem0 import Memory


def build_memory(collection_name: str, qdrant_path: str, embedder_model: str = "multi-qa-MiniLM-L6-cos-v1",
                  llm_model: str = "openai/gpt-4o", use_local_llm: bool = True,
                  local_llm_url: str = "http://127.0.0.1:8420/v1") -> Memory:
    if use_local_llm:
        llm_config = {"provider": "vllm", "config": {"model": "local-qwen2.5-7b-instruct",
                                                       "vllm_base_url": local_llm_url}}
    else:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY not set -- mem0's openai LLM provider auto-detects it "
                              "from the environment; export it before calling build_memory(use_local_llm=False).")
        llm_config = {"provider": "openai", "config": {"model": llm_model}}

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "path": qdrant_path,
                "embedding_model_dims": 384,  # multi-qa-MiniLM-L6-cos-v1's real output dim
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embedder_model},
        },
        "llm": llm_config,
    }
    return Memory.from_config(config)


def add_turn(memory: Memory, conversation_id: str, speaker: str, text: str, date_prefix: str) -> dict:
    """date_prefix: e.g. '[8 May 2023, 1:56 pm]' -- see module docstring
    on why this textual workaround exists instead of a real timestamp
    parameter."""
    content = f"{date_prefix} {speaker}: {text}"
    return memory.add(content, user_id=conversation_id, infer=True)


def search(memory: Memory, conversation_id: str, query: str, top_k: int = 10) -> list[str]:
    """Returns the retrieved memory TEXTS only (matching how
    scripts/run_downstream_qa_eval.py's own policies hand a text block to
    the shared grounded-QA prompt) -- the final answer is generated by
    OUR OWN pipeline's answer_question, not Mem0's, so the only variable
    under test is Mem0's memory selection, not its answer generation.
    Same isolation principle REMem itself used for its own baselines
    ("we retrieve the top-10 of their processed chunks")."""
    result = memory.search(query, filters={"user_id": conversation_id}, top_k=top_k)
    hits = result.get("results", result) if isinstance(result, dict) else result
    return [h["memory"] for h in hits]
