# Evidence retention: LongMemEval

`ours` retains 690/734 memories (94.0%) across all conversations.

| Policy | Evidence retention rate | N covered QA pairs |
|---|---|---|
| no_forget | 1.0000 | 92 |
| fifo | 1.0000 | 92 |
| lru | 1.0000 | 92 |
| ours | 0.9783 | 92 |
| ours_utility | 0.9891 | 92 |
| ours_combo | 0.9783 | 92 |

## `ours`-evicted evidence-bearing memories, by mechanism

| Mechanism | Count | % of evicted evidence memories |
|---|---|---|
| ttl_only | 17 | 100.0% |
| action_only | 0 | 0.0% |
| both | 0 | 0.0% |
| neither(?) | 0 | 0.0% |

## TTL-calibration gap

For each evidence-bearing memory evicted under `ours`, `predicted_ttl_days` (Lifetime head) vs. actual age at the point it was still needed as evidence:

| | mean | median |
|---|---|---|
| predicted_ttl_days (evicted evidence) | 13.4 | 9.0 |
| actual age needed | 63.0 | 21.6 |
| shortfall (age - predicted_ttl, days) | 49.6 | 8.8 |
| predicted_ttl_days (surviving evidence, for reference) | 54.0 | 37.5 |