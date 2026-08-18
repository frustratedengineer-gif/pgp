"""bootstrap_paired_mean_diff backs every Week-6 downstream EM/F1 claim
(week6_downstream_significance.md) -- if this is wrong, so are those."""
import numpy as np
import pytest

from memorylife.evaluation.downstream_significance import bootstrap_paired_mean_diff


def test_identical_arrays_are_statistically_indistinguishable():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    r = bootstrap_paired_mean_diff(values, values, n_boot=500, seed=1)
    assert r["mean_diff"] == 0.0
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["p_value_one_sided"] == 1.0  # a never beats b when a == b everywhere


def test_a_reliably_beating_b_gives_a_ci_that_excludes_zero():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, size=200).astype(float)
    a = np.clip(b + 0.3, 0, 1)  # a >= b for every single paired question
    r = bootstrap_paired_mean_diff(a, b, n_boot=1000, seed=42)
    assert r["mean_diff"] > 0
    assert r["ci_low"] > 0  # doesn't straddle zero -> "reliably beats it"
    assert r["p_value_one_sided"] == 0.0


def test_mismatched_lengths_raise_instead_of_silently_misaligning():
    with pytest.raises(AssertionError):
        bootstrap_paired_mean_diff([1.0, 0.0], [1.0], n_boot=10)


def test_same_seed_is_reproducible():
    a = [0.8, 0.6, 0.9, 0.4, 0.7]
    b = [0.5, 0.5, 0.5, 0.5, 0.5]
    r1 = bootstrap_paired_mean_diff(a, b, n_boot=200, seed=7)
    r2 = bootstrap_paired_mean_diff(a, b, n_boot=200, seed=7)
    assert r1 == r2
