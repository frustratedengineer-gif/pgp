TTL quantile (S(t) cutoff): 0.2 (0.5=median). `ours` retains 2315/2536 memories (91.3%) across all LoCoMo conversations.

| Policy | Evidence retention rate | N covered QA pairs |
|---|---|---|
| no_forget | 1.0000 | 1304 |
| fifo | 0.9225 | 1304 |
| lru | 0.9517 | 1304 |
| ours | 0.9080 | 1304 |
| ours_utility | 0.9663 | 1304 |
| ours_combo | 0.9363 | 1304 |

## `ours`-evicted evidence-bearing memories, by mechanism

| Mechanism | Count | % of evicted evidence memories |
|---|---|---|
| ttl_only | 271 | 100.0% |
| action_only | 0 | 0.0% |
| both | 0 | 0.0% |
| neither(?) | 0 | 0.0% |

## TTL-calibration gap (only meaningful mechanism per the table above)

For each evidence-bearing memory evicted under `ours`, `predicted_ttl_days` (Lifetime head) vs. actual age at the point it was still needed as evidence:

| | mean | median |
|---|---|---|
| predicted_ttl_days (evicted evidence) | 170.9 | 168.1 |
| actual age needed | 204.3 | 209.2 |
| shortfall (age - predicted_ttl, days) | 33.4 | 22.5 |
| predicted_ttl_days (surviving evidence, for reference) | 232.0 | 181.7 |