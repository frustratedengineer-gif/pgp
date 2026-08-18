# Oracle vs. no_forget: bootstrap significance (follow-up to week6_oracle_fullcontext.md)

n_boot=10000, seed=42, paired on (conversation_id, question) -- only questions where an oracle answer could be built (i.e. evidence coverage exists) are included.

| Benchmark | Comparison (a vs b) | N | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |
|---|---|---|---|---|---|---|
| locomo | oracle vs no_forget | n=107 | +0.0654 [+0.0187, +0.1215] | p=0.0073 | +0.1416 [+0.0829, +0.2033] | p=0.0000 |
| longmemeval | oracle vs no_forget | n=22 | +0.0455 [+0.0000, +0.1364] | p=0.3636 | +0.0152 [+0.0000, +0.0455] | p=0.3636 |