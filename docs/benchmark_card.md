# Benchmark Card: MemoryLifeBench

## Summary

MemoryLifeBench reframes memory management for personal AI assistants as a
**time-to-event (survival analysis) problem**: instead of "should I store
this memory?", the label is "how long will this memory remain useful?".

## Sources and composition

| Source | Train | Val | Test | What it is |
|---|---|---|---|---|
| synthetic | 3199 | 256 | 265 | LLM-generated multi-session dialogues with facts injected and probe questions scheduled by construction (exact, provable lifetimes) |
| longmemeval | 3162 | 359 | 375 | Candidate memories extracted from LongMemEval multi-session dialogues |
| locomo | 1936 | 324 | 276 | Candidate memories extracted from LoCoMo multi-session dialogues |
| **Total** | **8297** | **939** | **916** | |

Splits are by `conversation_id` (568 / 71 / 71 conversations) -- verified
disjoint, so no conversation's memories leak across splits
(`memorylife.data.splits.assert_no_leakage`).

## Label definition

For each memory record:

- **T (`duration_days`)**: time from `injected_at` (when the statement was
  made) to the reference event.
- **delta (`event_observed`)**: 1 if the event was actually observed
  (`censored=False`), 0 if censored.

## Censoring convention

A memory is **censored** (`censored=True`, `invalidated_at=None`) when we
never observed it being invalidated, updated, contradicted, or naturally
expiring within the available data. Three cases, all handled in
`src/memorylife/data/censoring.py`:

1. **`censored=False`** (event observed): `T = invalidated_at - injected_at`.
   `lifecycle_event` is one of `update`, `contradiction`, `natural_expiry`,
   `observed_usage`.
2. **`censored=True`, has probes** (synthetic only): `T = max(probe_at) -
   injected_at`. The last scheduled probe is the last point we actively
   checked whether the memory still held.
3. **`censored=True`, no probes** (real: longmemeval/locomo,
   `lifecycle_event="no_usage_observed"`): `T = conversation_max_timestamp -
   injected_at`, where `conversation_max_timestamp` is the latest timestamp
   (any `injected_at`/`invalidated_at`/`probe_at`) seen anywhere in the same
   `conversation_id`. This is **administrative censoring**: the last point
   we observed that conversation at all, not a per-record property. Real
   conversations have a median of 8 memory records spanning a median 17.5
   days (up to 298 days), so this is a meaningfully informative proxy, not
   a single global cutoff.

Durations are floored at 0.01 days to stay strictly positive for the
survival loss.

**This is a judgment call, not a given fact** -- case 3 has no ground-truth
alternative available in the source data. It should be stated explicitly
in the paper's methodology section, and is a candidate for a sensitivity
analysis (does the C-index ranking change under a different censoring-time
choice for this subset?).

## Class balance

| Split | Event rate |
|---|---|
| train | 3087 / 8297 = 37.2% |
| val | 333 / 939 = 35.5% |
| test | 335 / 916 = 36.6% |

By source (train): locomo 48.8%, synthetic 43.1%, longmemeval 24.1% --
longmemeval's low event rate reflects that most of its records are
`no_usage_observed` (never referenced again in the transcript).

## Known gaps

- The dialogue -> candidate-memory extraction pipeline that *produced*
  `data/raw/*.jsonl` is not yet in this repository (see `data/README.md`).
  A reviewer will ask for it.
- No inter-annotator agreement / human validation of the extracted labels
  has been done -- `docs/annotation_guidelines.md` is empty because no
  human annotation step exists yet; all labels are programmatically
  derived. If any manual QA is added later, document it there.
- ~~License terms for redistributing LongMemEval/LoCoMo-derived text have
  not been checked~~ -- resolved Week 6: see `LICENSE-DATA` (CC BY-NC 4.0,
  LoCoMo's own license is the binding constraint). `data/raw/` remains
  gitignored regardless (see `data/README.md`).
