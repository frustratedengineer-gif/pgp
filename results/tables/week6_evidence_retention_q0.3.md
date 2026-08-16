TTL quantile (S(t) cutoff): 0.3 (0.5=median). `ours` retains 2188/2536 memories (86.3%) across all LoCoMo conversations.

| Policy | Evidence retention rate | N covered QA pairs |
|---|---|---|
| no_forget | 1.0000 | 1304 |
| fifo | 0.8742 | 1304 |
| lru | 0.9118 | 1304 |
| ours | 0.8505 | 1304 |

## `ours`-evicted evidence-bearing memories, by mechanism

| Mechanism | Count | % of evicted evidence memories |
|---|---|---|
| ttl_only | 421 | 100.0% |
| action_only | 0 | 0.0% |
| both | 0 | 0.0% |
| neither(?) | 0 | 0.0% |

## TTL-calibration gap (only meaningful mechanism per the table above)

For each evidence-bearing memory evicted under `ours`, `predicted_ttl_days` (Lifetime head) vs. actual age at the point it was still needed as evidence:

| | mean | median |
|---|---|---|
| predicted_ttl_days (evicted evidence) | 151.8 | 148.8 |
| actual age needed | 190.7 | 194.8 |
| shortfall (age - predicted_ttl, days) | 38.8 | 26.7 |
| predicted_ttl_days (surviving evidence, for reference) | 198.9 | 158.4 |