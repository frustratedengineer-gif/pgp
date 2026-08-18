"""quantile_ttl_days (src/memorylife/inference/pipeline.py) is the Week-6
Fix #1: the eviction cutoff used to be hardcoded at the survival curve's
median (S(t)=0.5), a coin-flip threshold by construction. This proves the
configurable quantile actually behaves as documented: a lower quantile
(a later point on the same decaying curve) must push the cutoff later,
never earlier, and a curve that never decays past the quantile must fall
back to MAX_SURVIVAL_TTL_DAYS instead of an undefined value."""
import pandas as pd

from memorylife.inference.pipeline import MAX_SURVIVAL_TTL_DAYS, quantile_ttl_days

TIMES = [1, 5, 10, 20, 50]
DECAYING_CURVE = [1.0, 0.9, 0.6, 0.3, 0.1]  # crosses 0.5 between t=5 and t=10, 0.2 between t=20 and t=50
FLAT_CURVE = [1.0, 0.95, 0.9, 0.85, 0.8]  # never drops below 0.5


def _surv_df(**columns):
    return pd.DataFrame(columns, index=TIMES)


def test_median_cutoff_is_the_last_time_survival_stayed_at_or_above_it():
    surv_df = _surv_df(a=DECAYING_CURVE)
    ttl = quantile_ttl_days(surv_df, quantile=0.5)
    assert ttl[0] == 10  # S(10)=0.6 >= 0.5, S(20)=0.3 < 0.5 -> cutoff is t=10


def test_lower_quantile_pushes_the_cutoff_later_on_the_same_curve():
    surv_df = _surv_df(a=DECAYING_CURVE)
    ttl_median = quantile_ttl_days(surv_df, quantile=0.5)[0]
    ttl_q20 = quantile_ttl_days(surv_df, quantile=0.2)[0]
    assert ttl_q20 > ttl_median  # the documented Fix #1 behavior


def test_curve_that_never_decays_past_quantile_gets_the_max_ttl_cap():
    surv_df = _surv_df(a=FLAT_CURVE)
    ttl = quantile_ttl_days(surv_df, quantile=0.5)
    assert ttl[0] == MAX_SURVIVAL_TTL_DAYS


def test_multiple_columns_are_scored_independently():
    surv_df = _surv_df(decaying=DECAYING_CURVE, flat=FLAT_CURVE)
    ttl = quantile_ttl_days(surv_df, quantile=0.5)
    assert ttl[0] == 10
    assert ttl[1] == MAX_SURVIVAL_TTL_DAYS
