TTL quantile (S(t) cutoff): 0.05 (0.5=median). `ours` retains 2462/2536 memories (97.1%) across all LoCoMo conversations.

| Policy | Evidence retention rate | N covered QA pairs |
|---|---|---|
| no_forget | 1.0000 | 1304 |
| fifo | 0.9762 | 1304 |
| lru | 0.9778 | 1304 |
| ours | 0.9716 | 1304 |

## `ours`-evicted evidence-bearing memories, by mechanism

| Mechanism | Count | % of evicted evidence memories |
|---|---|---|
| ttl_only | 82 | 100.0% |
| action_only | 0 | 0.0% |
| both | 0 | 0.0% |
| neither(?) | 0 | 0.0% |

## TTL-calibration gap (only meaningful mechanism per the table above)

For each evidence-bearing memory evicted under `ours`, `predicted_ttl_days` (Lifetime head) vs. actual age at the point it was still needed as evidence:

| | mean | median |
|---|---|---|
| predicted_ttl_days (evicted evidence) | 192.5 | 191.5 |
| actual age needed | 223.8 | 226.6 |
| shortfall (age - predicted_ttl, days) | 31.2 | 26.5 |
| predicted_ttl_days (surviving evidence, for reference) | 357.1 | 217.9 |