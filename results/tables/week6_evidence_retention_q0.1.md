TTL quantile (S(t) cutoff): 0.1 (0.5=median). `ours` retains 2415/2536 memories (95.2%) across all LoCoMo conversations.

| Policy | Evidence retention rate | N covered QA pairs |
|---|---|---|
| no_forget | 1.0000 | 1304 |
| fifo | 0.9670 | 1304 |
| lru | 0.9709 | 1304 |
| ours | 0.9502 | 1304 |

## `ours`-evicted evidence-bearing memories, by mechanism

| Mechanism | Count | % of evicted evidence memories |
|---|---|---|
| ttl_only | 147 | 100.0% |
| action_only | 0 | 0.0% |
| both | 0 | 0.0% |
| neither(?) | 0 | 0.0% |

## TTL-calibration gap (only meaningful mechanism per the table above)

For each evidence-bearing memory evicted under `ours`, `predicted_ttl_days` (Lifetime head) vs. actual age at the point it was still needed as evidence:

| | mean | median |
|---|---|---|
| predicted_ttl_days (evicted evidence) | 189.4 | 180.2 |
| actual age needed | 218.4 | 226.6 |
| shortfall (age - predicted_ttl, days) | 29.0 | 26.4 |
| predicted_ttl_days (surviving evidence, for reference) | 293.7 | 201.0 |