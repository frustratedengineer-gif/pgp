"""
Bootstrap significance testing for the Week-6 downstream QA comparisons
(EM/F1 across forgetting policies) -- the same paired-bootstrap method as
`evaluation/significance.py` uses for the Week-3/4 C-index claims, applied
here to the mean-EM/mean-F1 claims in `week6_downstream_qa*.md` and
`week6_ranked_eviction_sweep.md`. Those claims were reported as raw means
with no confidence interval or significance test -- this closes that gap.

Paired, not independent: every comparison in Week 6 was run on the SAME
set of questions for every policy (matched storage budget, matched
question sample), so the same resampled question indices are used to
recompute both policies' mean metric in each bootstrap replicate.
"""
import numpy as np


def bootstrap_paired_mean_diff(values_a, values_b, n_boot: int = 1000, seed: int = 42) -> dict:
    """CI + one-sided p-value for mean(values_a) - mean(values_b), where
    values_a[i]/values_b[i] are the SAME question's per-item metric (EM or
    F1) under two different policies. p_value_one_sided is the fraction of
    bootstrap replicates where a did NOT beat b (a - b <= 0); small means
    "reliably beats it". A CI that straddles 0 (ci_low < 0 < ci_high)
    means "statistically indistinguishable" -- the relevant read for an
    "ours_utility reaches the no_forget ceiling" claim."""
    rng = np.random.default_rng(seed)
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    assert len(values_a) == len(values_b), "paired arrays must be the same length (same questions)"
    n = len(values_a)

    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = values_a[idx].mean(axis=1) - values_b[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value_one_sided": float((diffs <= 0).mean()),
        "n": n, "n_boot": n_boot,
    }
