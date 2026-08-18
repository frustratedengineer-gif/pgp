"""format_report (scripts/diagnose_eviction_evidence.py) turns an Aggregator's
raw counters into the numbers actually quoted in
results/tables/week6_evidence_retention*.md (retention rate per policy,
eviction-mechanism attribution, TTL-calibration gap) -- this checks the
arithmetic directly against a hand-computed example instead of trusting
the table by eye."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from diagnose_eviction_evidence import POLICIES, Aggregator, format_report  # noqa: E402


def _populated_agg():
    agg = Aggregator()
    agg.capacities = [3]
    agg.total_objects = [5]
    agg.retained = {
        "no_forget": [1, 1], "fifo": [1, 0], "lru": [1, 1],
        "ours": [0, 1], "ours_utility": [1, 1], "ours_combo": [1, 1],
    }
    agg.eviction_mechanism = Counter({"ttl_only": 2, "action_only": 1})
    agg.evicted_ttls = [10.0, 20.0]
    agg.evicted_ages = [15.0, 25.0]
    agg.survived_ttls = [30.0]
    return agg


def test_all_policies_present_with_correct_retention_rate():
    report = format_report(_populated_agg(), "Test Title")
    assert "# Test Title" in report
    assert "| no_forget | 1.0000 | 2 |" in report  # mean([1,1])
    assert "| ours | 0.5000 | 2 |" in report  # mean([0,1])
    assert "| fifo | 0.5000 | 2 |" in report


def test_storage_fraction_uses_summed_capacities_over_summed_totals():
    report = format_report(_populated_agg(), "Test Title")
    assert "3/5" in report
    assert "60.0%" in report  # 3/5


def test_eviction_mechanism_percentages_are_of_the_mechanism_total_not_all_evictions():
    report = format_report(_populated_agg(), "Test Title")
    # 2 ttl_only + 1 action_only + 0 both + 0 neither(?) = 3 total
    assert "| ttl_only | 2 | 66.7% |" in report
    assert "| action_only | 1 | 33.3% |" in report
    assert "| both | 0 | 0.0% |" in report


def test_ttl_calibration_shortfall_is_age_minus_predicted_ttl():
    report = format_report(_populated_agg(), "Test Title")
    # shortfall = [15-10, 25-20] = [5, 5] -> mean=median=5.0
    assert "shortfall (age - predicted_ttl, days) | 5.0 | 5.0 |" in report


def test_empty_evicted_evidence_is_reported_explicitly_not_as_a_crash():
    agg = Aggregator()
    agg.capacities = [5]
    agg.total_objects = [5]
    for p in POLICIES:
        agg.retained[p] = [1]
    report = format_report(agg, "No Evictions")
    assert "(no evicted evidence-bearing memories in this sample)" in report
