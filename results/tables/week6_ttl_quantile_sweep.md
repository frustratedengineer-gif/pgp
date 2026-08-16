# TTL-quantile sweep: does moving the eviction cutoff off the median fix evidence retention?

Free, local, no-LLM-calls diagnostic (`scripts/diagnose_eviction_evidence.py
--ttl-quantile Q`) -- set arithmetic only, against the same gold evidence
used to measure `results/tables/week6_evidence_retention.md`. `fifo`/`lru`
capacity is re-matched to `ours`'s own count at each quantile, so this is
still an apples-to-apples storage-budget comparison at every row.

`Q` = the survival probability `S(t)` the TTL cutoff crosses (`0.5` =
median, Week 6's original default; lower `Q` pushes the cutoff later,
i.e. keeps memories longer -- see `quantile_ttl_days` in
`src/memorylife/inference/pipeline.py`).

| Q (S(t) cutoff) | `ours` storage kept | `ours` evidence retention | `fifo` retention (matched capacity) | `lru` retention (matched capacity) |
|---|---|---|---|---|
| 0.5 (median, original) | 70.9% (1797/2536) | 0.6687 | 0.7247 | 0.7676 |
| 0.3 | 86.3% (2188/2536) | 0.8505 | 0.8742 | 0.9118 |
| 0.2 | 91.3% (2315/2536) | 0.9080 | 0.9225 | 0.9517 |
| 0.1 | 95.2% (2415/2536) | 0.9502 | 0.9670 | 0.9709 |
| 0.05 | 97.1% (2462/2536) | 0.9716 | 0.9762 | 0.9778 |

(`no_forget` ceiling is 1.0000 at every row, 1304 covered LoCoMo QA pairs.)

## Reading this

1. **The quantile fix works, in the sense predicted**: moving the TTL
   cutoff from the median (Q=0.5) to a more conservative quantile
   monotonically and substantially closes the evidence-retention gap --
   from 0.6687 to 0.9716 as Q -> 0.05, confirming the root cause
   diagnosed in `week6_evidence_retention.md` (median-as-hard-cutoff is a
   coin-flip threshold, not a model-quality problem).
2. **It does not fully close the gap against fifo/lru at matched
   capacity, at any Q tested.** At every single row, `fifo` and
   especially `lru` retain evidence at least as well as `ours`, often
   slightly better (e.g. Q=0.2: ours 0.9080 vs. lru 0.9517). As Q -> 0
   all three policies' capacities approach the full store and their
   retention rates converge toward the 1.0 ceiling, which mechanically
   shrinks the gap -- but the ranking (lru > fifo > ours) holds at every
   Q, so this is a real, if now much smaller, second-order effect, not
   just noise.
3. **Working hypothesis for the residual gap**: recency and predicted
   remaining-lifetime are different selection criteria, and LoCoMo's QA
   evidence may itself be recency-skewed (more questions about recently
   mentioned facts than old ones) -- which would mechanically favor
   fifo/lru's "keep the newest N" / "keep the most-recently-touched N"
   selection over a lifetime-ranking-based selection, independent of
   whether the lifetime ranking itself is accurate. Not yet verified;
   natural follow-up is measuring evidence-dia_id recency-within-
   conversation directly. Noted as a limitation, not chased further here
   given scope.

## Recommended operating point

**Q=0.2** is a reasonable default to carry into a real (paid, LLM-scored)
EM/F1 re-run: it recovers the large majority of the gap (0.9080 vs. the
0.6687 original) at a moderate storage cost (91.3% vs. the original
70.9%), without going so low-Q that the comparison degenerates toward
`no_forget` and stops testing the forgetting policy at all. Not yet run
through the paid downstream QA eval (`scripts/run_downstream_qa_eval.py`,
real GPT-4o calls) -- that is the next step to confirm the retention-rate
improvement actually translates to EM/F1, rather than assuming it does.
