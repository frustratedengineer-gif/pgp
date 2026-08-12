"""
Bootstrap significance testing for C-index comparisons.

Two distinct sources of uncertainty are tracked separately in this repo and
should not be conflated:
  - training-time variance across random seeds -> results/tables/week4_multiseed_results.md
  - sampling uncertainty from the finite test set, for a FIXED trained model
    -> this module, results/tables/week4_significance.md

`bootstrap_paired_diff` is the standard paired-bootstrap test for ranking
metrics: the same resampled record indices are used to recompute both
methods' C-index in each replicate, so the comparison is paired (not two
independent CIs eyeballed against each other).
"""
import numpy as np

from memorylife.evaluation.survival_metrics import c_index


def bootstrap_c_index(durations, scores, events, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    durations = np.asarray(durations)
    scores = np.asarray(scores)
    events = np.asarray(events)
    n = len(durations)

    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot.append(c_index(durations[idx], scores[idx], events[idx]))
        except ZeroDivisionError:
            continue  # resample had no comparable pairs, vanishingly rare at n~900
    boot = np.array(boot)
    return {
        "mean": float(boot.mean()),
        "std": float(boot.std()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_boot": int(len(boot)),
    }


def bootstrap_paired_diff(durations, scores_a, scores_b, events, n_boot: int = 1000, seed: int = 42) -> dict:
    """CI + one-sided p-value for c_index(a) - c_index(b), e.g. a="our model"
    b="a baseline". p_value_one_sided is the fraction of bootstrap replicates
    where a did NOT beat b (a - b <= 0); small means "reliably beats it"."""
    rng = np.random.default_rng(seed)
    durations = np.asarray(durations)
    events = np.asarray(events)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    n = len(durations)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            ca = c_index(durations[idx], scores_a[idx], events[idx])
            cb = c_index(durations[idx], scores_b[idx], events[idx])
        except ZeroDivisionError:
            continue
        diffs.append(ca - cb)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value_one_sided": float((diffs <= 0).mean()),
        "n_boot": int(len(diffs)),
    }
