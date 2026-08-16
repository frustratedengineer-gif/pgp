# Week-6 downstream QA: bootstrap significance (paired, same-question resampling)

n_boot=10000, seed=42. `p (a<=b)` is the fraction of bootstrap
replicates where policy `a` did NOT beat policy `b` -- small means `a` reliably
beats `b`; a CI straddling 0 alongside p near 0.5 means statistically
indistinguishable (the relevant read for an "ours_utility matches the ceiling" claim).

## Fix #2: does utility-ranked eviction (`ours_utility`) really beat lru / match the ceiling?

| Benchmark | Comparison (a vs b) | N | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |
|---|---|---|---|---|---|---|
| locomo | ours_utility vs lru | n=120 | +0.0083 [+0.0000, +0.0250] | p=0.367 | -0.0048 [-0.0299, +0.0180] | p=0.648 |
| locomo | ours_utility vs fifo | n=120 | +0.0499 [+0.0167, +0.0917] | p=0.002 | +0.0656 [+0.0216, +0.1121] | p=0.001 |
| locomo | ours_utility vs ours | n=120 | +0.0415 [+0.0083, +0.0833] | p=0.006 | +0.0375 [+0.0124, +0.0682] | p=0.001 |
| locomo | ours_utility vs no_forget | n=120 | +0.0000 [+0.0000, +0.0000] | p=1.000 | -0.0008 [-0.0174, +0.0158] | p=0.559 |
| longmemeval | ours_utility vs lru | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 | +0.0269 [+0.0000, +0.0813] | p=0.130 |
| longmemeval | ours_utility vs fifo | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 | +0.0004 [-0.0100, +0.0114] | p=0.478 |
| longmemeval | ours_utility vs ours | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 | +0.0000 [+0.0000, +0.0000] | p=1.000 |
| longmemeval | ours_utility vs no_forget | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 | +0.0038 [+0.0000, +0.0114] | p=0.359 |

## Fix #1: does moving the TTL cutoff off the median significantly help `ours`?

| Benchmark | Comparison (a vs b) | N | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |
|---|---|---|---|---|---|---|
| locomo | ours@Q0.2 vs ours@Q0.5 | n=120 | +0.0166 [+0.0000, +0.0417] | p=0.137 | +0.0399 [+0.0056, +0.0785] | p=0.010 |
| locomo | ours@Q0.1 vs ours@Q0.5 | n=120 | +0.0249 [+0.0000, +0.0583] | p=0.048 | +0.0475 [+0.0103, +0.0887] | p=0.005 |
| locomo | ours@Q0.1 vs ours@Q0.2 | n=120 | +0.0083 [+0.0000, +0.0250] | p=0.367 | +0.0077 [-0.0014, +0.0198] | p=0.064 |
| longmemeval | ours@Q0.2 vs ours@Q0.5 | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 | +0.0000 [+0.0000, +0.0000] | p=1.000 |
| longmemeval | ours@Q0.1 vs ours@Q0.5 | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 | +0.0000 [+0.0000, +0.0000] | p=1.000 |
| longmemeval | ours@Q0.1 vs ours@Q0.2 | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 | +0.0000 [+0.0000, +0.0000] | p=1.000 |