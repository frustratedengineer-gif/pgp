# Bootstrap significance of the LLM-judge metric (closes a Limitations gap)

n_boot=10000, seed=42, paired same-question resampling -- identical method to
results/tables/week6_downstream_significance.md, applied to the `judge` field instead
of EM/F1. Source: results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json
(725 GPT-4o-judged predictions, already computed for Section 6.8 -- no new LLM calls).

| Benchmark | Comparison (a vs b) | N | Judge-score diff (a-b), 95% CI | p (a<=b) |
|---|---|---|---|---|
| locomo | ours_utility vs fifo | n=120 | +0.1671 [+0.0833, +0.2500] | p=0.000 |
| locomo | ours_utility vs ours | n=120 | +0.1505 [+0.0750, +0.2333] | p=0.000 |
| locomo | ours_utility vs no_forget | n=120 | +0.0167 [-0.0250, +0.0583] | p=0.269 |
| locomo | ours_utility vs lru | n=120 | +0.0085 [-0.0333, +0.0500] | p=0.421 |
| longmemeval | ours_utility vs fifo | n=25 | -0.0409 [-0.1200, +0.0000] | p=1.000 |
| longmemeval | ours_utility vs ours | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 |
| longmemeval | ours_utility vs no_forget | n=25 | +0.0000 [+0.0000, +0.0000] | p=1.000 |
| longmemeval | ours_utility vs lru | n=25 | +0.0793 [+0.0000, +0.2000] | p=0.130 |