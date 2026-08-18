# Refusal-precision bootstrap significance (follow-up to week6_refusal_eval.md)

n_boot=10000, seed=42, paired resampling over the shared 240-question set (120 answerable + 120 adversarial) per policy.

| Comparison (a vs b) | N | Precision diff (a-b), 95% CI | p (a<=b) |
|---|---|---|---|
| ours_utility vs ours | n=240 | +0.0776 [+0.0394, +0.1191] | p=0.000 |
| ours_utility vs fifo | n=240 | +0.0873 [+0.0460, +0.1315] | p=0.000 |
| no_forget vs ours | n=240 | +0.0697 [+0.0281, +0.1143] | p=0.001 |
| no_forget vs fifo | n=240 | +0.0794 [+0.0396, +0.1232] | p=0.000 |