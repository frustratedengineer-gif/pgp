# Data

## Sources

| Source | What it is | License note |
|---|---|---|
| `synthetic` | LLM-generated multi-session dialogues with facts injected and probe questions scheduled by us (Week 2) | Ours |
| `longmemeval` | Candidate memories extracted from the LongMemEval benchmark's multi-session dialogues | Third-party (MIT-declared repo) |
| `locomo` | Candidate memories extracted from the LoCoMo benchmark's multi-session dialogues | Third-party (CC BY-NC 4.0) |

See `LICENSE-DATA` at the repo root for the resolved combined license
(CC BY-NC 4.0, non-commercial only, driven by LoCoMo's own license) and
required attribution -- this table is a summary, not the authoritative
statement.

`data/raw/{train,val,test}.jsonl` are the pre-extracted, pre-labeled memory
records (one fact-statement per record, with `injected_at` / `invalidated_at`
/ `censored` / `probes` already resolved) — **not** the original LoCoMo /
LongMemEval dialogue dumps. See "Known gap" below.

## Known gap: dataset-generation code

**Correction found during a final consistency pass (this line was wrong,
possibly since early Week 6, and never caught until now):** the claim
that `src/memorylife/data/build_benchmark.py` "is still an empty stub"
is **false**. That file is 100 lines of real, working code
(`load_locomo_qa`, `load_longmemeval_qa`, `_dia_ids_of`, `evidence_coverage`,
`load_all_processed`) and is, in fact, the single most-imported module in
this project's entire Week-6 downstream-evaluation work --
`scripts/run_downstream_qa_eval.py`, `diagnose_eviction_evidence.py`,
`eval_refusal.py`, `eval_oracle_fullcontext.py`, `eval_mem0_baseline.py`,
`qualitative_examples.py`, `analyze_utility_signal.py`,
`categorize_error_sample.py`, and `breakdown_by_question_type.py` all
import from it directly. `scripts/build_benchmark.py` (37 lines) is a
thin CLI wrapper around it, not a separate implementation -- the actual
load/link/evidence-coverage logic lives in the `src/` module, not the
`scripts/` one; a previous version of this note attributed it to the
wrong file.

What genuinely IS still missing, and is the real, narrower gap: the
dialogue -> candidate-memory EXTRACTION pipeline (extracting fact
statements from raw conversation turns, determining `lifecycle_event`,
generating the synthetic conversations for Week 2) that ORIGINALLY
produced `data/raw/*.jsonl` is not in this repository, and
`src/memorylife/data/build_benchmark.py`'s own docstring says so
directly -- it explicitly states it is "NOT the original dialogue ->
candidate-memory extraction pipeline," and does a narrower job instead:
reads already-published QA pairs and dialogue text from the raw
benchmark files and links them back to our already-extracted
`data/raw/*.jsonl` records via `conversation_id` (matches LoCoMo's
`sample_id`) and `evidence_dia_id` (matches LoCoMo's per-turn `dia_id` /
LongMemEval's `haystack_session_ids`). `data/download.sh` is a
placeholder for the same underlying reason (no extraction pipeline to
run it against).

## Raw benchmark files (not committed, not derived -- placed here manually)

`data/raw/locomo10.json` (10 conversations, ~199 QA pairs each) and
`data/raw/longmemeval_s_cleaned.json` (500 QA pairs, up to 53 haystack
sessions each) are the original published benchmark files -- full dialogue
transcripts plus reference question/answer pairs, used by
`scripts/run_downstream_qa_eval.py` for the downstream memory-system
comparison (`baselines/README.md`). **Gitignored** under the same
`data/raw/*` rule as everything else in this directory -- these are
third-party benchmark data (see the license note in the Sources table
above), large (LongMemEval's file is ~277MB), and not something this repo
redistributes. If you need them: LoCoMo is published by the LoCoMo paper's
authors; LongMemEval by its authors -- check each benchmark's own license
before use. `conversation_id` in `data/raw/*.jsonl` links directly to
LoCoMo's `sample_id` field and to `lme_<question_id>` for LongMemEval,
confirmed by direct lookup (not assumed) before building the eval harness.

## What's derived vs. what's committed

- `data/raw/` — the three files above. **Gitignored** (not committed):
  regenerate via `data/download.sh` once it exists, or copy them back from
  wherever they currently live.
- `data/processed/` — `{split}_survival.jsonl`, adds `duration_days` /
  `event_observed` / `censor_reason` to every record. **Gitignored**;
  regenerate with `python scripts/preprocess.py`.
- `data/splits/{train,val,test}_ids.txt` — **committed**. The exact
  `memory_id` membership of each split, so splits are reproducible without
  redistributing the data itself. Verified conversation-disjoint (no
  `conversation_id` appears in more than one split) by
  `memorylife.data.splits.assert_no_leakage`, called from
  `scripts/preprocess.py`.
- `data/samples/memories_sample.jsonl` — **committed**. 60 records (20 per
  source) from the val split, so the repo has something to run against out
  of the box without the full dataset. There are no separate
  `dialogues_sample.jsonl` / `lifetime_events_sample.jsonl` files (as a
  generic template might expect) because this schema doesn't keep raw
  dialogue transcripts separately from extracted memories — see the "known
  gap" above.
- `data/checksums.sha256` — sha256 of every raw/processed/sample file as of
  the Week-3 results, for integrity checking once the real files are back
  in place.

## Record schema

See `src/memorylife/data/schema.py` for the authoritative field list, and
`docs/benchmark_card.md` for the label-definition writeup (what
`duration_days` / `event_observed` mean, and the censoring-time convention
for records with `censored=True`).

## Splits

| Split | N | Conversations | Event rate |
|---|---|---|---|
| train | 8297 | 568 | ~35% (varies by source, see benchmark card) |
| val | 939 | 71 | ~36% |
| test | 916 | 71 | ~37% |

Split by `conversation_id`, not by record — no conversation's memories are
split across train/val/test.
