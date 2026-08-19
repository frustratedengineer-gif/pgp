# Oracle vs. full_context vs. no_forget: three-way matched comparison (LoCoMo)

Follow-up to week6_oracle_fullcontext.md / week6_oracle_significance.md -- those tables report each policy's own full sample (oracle N=107, full_context N=30, no_forget N=120, all different because `oracle` requires evidence coverage and `full_context` was only run for a capped 3-questions-per-conversation sample). This is the SAME 3 policies restricted to the exact questions all three have a scored row for, so mean/significance comparisons here are genuinely paired, not just three separate marginal means.

**N = 22** questions where oracle, full_context, and no_forget were all scored.

| Policy | N | Mean EM | Mean F1 |
|---|---|---|---|
| oracle | 22 | 0.2273 | 0.4557 |
| full_context | 22 | 0.1364 | 0.3495 |
| no_forget | 22 | 0.1818 | 0.2966 |

## Pairwise bootstrap significance (n_boot=10000, seed=42)

| Comparison (a vs b) | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |
|---|---|---|---|---|
| oracle vs full_context | +0.0917 [+0.0000, +0.2273] | p=0.1221 | +0.1069 [-0.0246, +0.2595] | p=0.0609 |
| full_context vs no_forget | -0.0451 [-0.1364, +0.0000] | p=1.0000 | +0.0529 [+0.0000, +0.1241] | p=0.0409 |
| oracle vs no_forget | +0.0466 [-0.0909, +0.1818] | p=0.3815 | +0.1598 [+0.0385, +0.3063] | p=0.0020 |