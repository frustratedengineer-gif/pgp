# Mem0 baseline: bootstrap significance vs. this project's own policies

n_boot=10000, seed=42, paired on (conversation_id, question) -- same 120 LoCoMo questions as week6_mem0_baseline.md and week6_downstream_qa_q0.2_ranked_pilot.md.

| Comparison (a vs b) | N | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |
|---|---|---|---|---|---|
| mem0 vs no_forget | n=120 | -0.0078 [-0.0583, +0.0417] | p=0.6840 | +0.0321 [-0.0247, +0.0899] | p=0.1324 |
| mem0 vs ours_utility | n=120 | -0.0078 [-0.0583, +0.0417] | p=0.6840 | +0.0329 [-0.0247, +0.0918] | p=0.1359 |
| mem0 vs lru | n=120 | +0.0005 [-0.0417, +0.0500] | p=0.5607 | +0.0281 [-0.0312, +0.0880] | p=0.1765 |
| mem0 vs fifo | n=120 | +0.0421 [+0.0000, +0.0833] | p=0.0311 | +0.0984 [+0.0371, +0.1623] | p=0.0008 |
| mem0 vs ours | n=120 | +0.0337 [+0.0000, +0.0750] | p=0.0623 | +0.0704 [+0.0142, +0.1298] | p=0.0053 |