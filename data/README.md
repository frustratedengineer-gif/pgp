# Data

## Sources

| Source | What it is | License note |
|---|---|---|
| `synthetic` | LLM-generated multi-session dialogues with facts injected and probe questions scheduled by us (Week 2) | Ours; safe to redistribute |
| `longmemeval` | Candidate memories extracted from the LongMemEval benchmark's multi-session dialogues | Third-party benchmark; check LongMemEval's license before redistributing full raw text |
| `locomo` | Candidate memories extracted from the LoCoMo benchmark's multi-session dialogues | Third-party benchmark; check LoCoMo's license before redistributing full raw text |

`data/raw/{train,val,test}.jsonl` are the pre-extracted, pre-labeled memory
records (one fact-statement per record, with `injected_at` / `invalidated_at`
/ `censored` / `probes` already resolved) — **not** the original LoCoMo /
LongMemEval dialogue dumps. See "Known gap" below.

## Known gap: dataset-generation code

The dialogue -> candidate-memory extraction pipeline (loading raw LoCoMo /
LongMemEval dialogues, extracting fact statements, determining
`lifecycle_event`, generating the synthetic conversations for Week 2) is
**not yet in this repository** — `src/memorylife/data/build_benchmark.py`
and `scripts/build_benchmark.py` are still empty stubs. `data/raw/*.jsonl`
arrived already built. Reviewers will ask for this; it needs to be written
and committed before submission. `data/download.sh` is a placeholder for
the same reason — see the script for what it should eventually do.

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
