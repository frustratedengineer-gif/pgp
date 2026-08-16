# Ranked eviction (utility-head top-N) vs. threshold eviction: free evidence-retention sweep

Same free, no-LLM-calls diagnostic as `week6_ttl_quantile_sweep.md`
(`scripts/diagnose_eviction_evidence.py`), extended with two new eviction
policies added to `scripts/run_downstream_qa_eval.final_active_ids`:

- **`ours_utility`**: rank all memories by the already-trained
  Future-Utility head's `P(retrieved again)` (`obj.utility_prob`), keep
  the top-N matching `ours`'s own capacity at that quantile. Uses a
  signal the joint model already produces but that eviction never
  consulted before now (previously only used to rerank retrieval
  results, see `inference/pipeline.format_retrieved_block`).
- **`ours_combo`**: rank by `0.5 * utility_prob + 0.5 *
  remaining_life_fraction`, where `remaining_life_fraction` is a
  continuous (not hard-cutoff) version of the TTL signal --
  `1 - age_days/predicted_ttl_days`, clipped to [0, 1].

Motivation: `ours` (original) makes an independent per-memory yes/no
threshold decision; `fifo`/`lru` always keep their best-N by a ranked
score. A threshold can waste capacity on a mediocre item whose score
happens to land on the "keep" side while a genuinely important one with
noisier scoring falls just short. `ours_utility`/`ours_combo` test
whether switching to a ranked top-N selection -- the same *structure*
fifo/lru use, just with a trained score instead of a heuristic one --
closes the residual gap `week6_ttl_quantile_sweep.md` left open.

| TTL quantile | fifo | lru | ours (threshold) | ours_combo (ranked, blended) | **ours_utility (ranked, pure)** |
|---|---|---|---|---|---|
| 0.5 (original) | 0.7247 | 0.7676 | 0.6687 | 0.7316 | **0.8765** |
| 0.2 | 0.9225 | 0.9517 | 0.9080 | 0.9363 | **0.9663** |
| 0.1 | 0.9670 | 0.9709 | 0.9502 | 0.9716 | **0.9877** |

(`no_forget` ceiling = 1.0000 at every row, 1304 covered LoCoMo QA pairs;
`ours`'s TTL-quantile choice still sets the shared capacity all policies
are matched to at each row.)

## Reading this

`ours_utility` beats **every** baseline, including `lru`, at **every**
quantile tested -- including at Q=0.5, the original unfixed setting,
where it recovers evidence retention from 0.6687 to 0.8765 (+0.208)
purely by changing the SELECTION MECHANISM (ranked top-N vs. threshold),
without touching the TTL-quantile fix at all. Combining it with the
quantile fix pushes it to 0.9877 at Q=0.1 -- 98.8% of the achievable
ceiling.

`ours_combo` is a real improvement over `ours` too, but is consistently
and clearly worse than pure `ours_utility` at every quantile -- the
utility head's signal is doing essentially all of the work; blending in
the continuous TTL signal dilutes rather than helps here. Worth noting
for the write-up as a "simpler was better" result, consistent with the
Week-5 finding that concat fusion beat the more sophisticated gated
fusion.

## Not yet done

This is still a free, LLM-call-free proxy metric. Real EM/F1 confirmation
(same matched-question methodology as `week6_downstream_qa_q0.5_pilot_control.md`
/ `..._q0.2_pilot.md` / `..._q0.1_pilot.md`) has not been run yet for
`ours_utility`. Given how large and consistent this effect is on the
proxy metric, it is a strong candidate for one more paid confirmation
run.
