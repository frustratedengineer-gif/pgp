| Policy | Evidence retention rate | N covered QA pairs |
|---|---|---|
| no_forget | 1.0000 | 1304 |
| fifo | 0.7247 | 1304 |
| lru | 0.7676 | 1304 |
| ours | 0.6687 | 1304 |

## `ours`-evicted evidence-bearing memories, by mechanism

| Mechanism | Count | % of evicted evidence memories |
|---|---|---|
| ttl_only | 903 | 100.0% |
| action_only | 0 | 0.0% |
| both | 0 | 0.0% |
| neither(?) | 0 | 0.0% |

## TTL-calibration gap (only meaningful mechanism per the table above)

For each evidence-bearing memory evicted under `ours`, `predicted_ttl_days` (Lifetime head) vs. actual age at the point it was still needed as evidence:

| | mean | median |
|---|---|---|
| predicted_ttl_days (evicted evidence) | 114.4 | 99.5 |
| actual age needed | 164.1 | 162.9 |
| shortfall (age - predicted_ttl, days) | 49.7 | 40.2 |
| predicted_ttl_days (surviving evidence, for reference) | 125.5 | 111.2 |