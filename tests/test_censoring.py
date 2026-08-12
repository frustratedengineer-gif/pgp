"""
The loss must handle censoring (delta=0) correctly -- this is the classic
failure mode in survival-analysis code, and the repo template calls this
file out explicitly (docs/repo_structure_reference.md). These tests pin
down src/memorylife/data/censoring.py + event_labeling.py's contract:

  - an observed event uses invalidated_at directly, event=1
  - a censored record WITH probes uses the last probe_at, event=0
  - a censored record WITHOUT probes (real data, no probes) uses
    administrative censoring: the latest timestamp seen anywhere in the
    same conversation_id
  - duration is floored at MIN_DURATION_DAYS, never zero or negative
"""
from memorylife.data.censoring import (
    MIN_DURATION_DAYS,
    conversation_max_timestamp,
    resolve_censoring_time,
)
from memorylife.data.event_labeling import compute_target, label_records
from memorylife.data.schema import VALID_CENSOR_REASONS, MemoryRecord


def _record(**overrides) -> dict:
    base = {
        "memory_id": "m1",
        "source": "synthetic",
        "conversation_id": "c1",
        "category": "test",
        "injected_at": "2026-01-01T00:00:00",
        "lifetime_is_exact": True,
        "text": "test memory",
        "lifecycle_event": "none",
        "censored": False,
        "invalidated_at": "2026-01-03T00:00:00",
        "probes": [],
    }
    base.update(overrides)
    return base


def test_observed_event_uses_invalidated_at_directly():
    rec = _record(censored=False, invalidated_at="2026-01-03T00:00:00")
    duration, event, reason = compute_target(rec, conv_max={})
    assert event == 1
    assert reason == "observed_event"
    assert duration == 2.0  # exactly 2 days between injected_at and invalidated_at


def test_censored_with_probes_uses_last_probe():
    rec = _record(
        censored=True, invalidated_at=None,
        probes=[{"probe_at": "2026-01-02T00:00:00", "expected_answer_source": "original"},
                {"probe_at": "2026-01-05T00:00:00", "expected_answer_source": "original"}],
    )
    duration, event, reason = compute_target(rec, conv_max={})
    assert event == 0
    assert reason == "censored_last_probe"
    assert duration == 4.0  # latest probe (Jan 5) is used, not the first (Jan 2)


def test_censored_without_probes_uses_conversation_administrative_censoring():
    records = [
        _record(memory_id="m1", conversation_id="c1", injected_at="2026-01-01T00:00:00",
                censored=True, invalidated_at=None, probes=[]),
        # a sibling record in the same conversation, observed later -- this
        # is what should set the administrative censoring time for m1
        _record(memory_id="m2", conversation_id="c1", injected_at="2026-01-01T00:00:00",
                censored=False, invalidated_at="2026-01-10T00:00:00"),
    ]
    conv_max = conversation_max_timestamp(records)
    duration, event, reason = compute_target(records[0], conv_max)
    assert event == 0
    assert reason == "censored_conversation_end"
    assert duration == 9.0  # censored at the conversation's latest timestamp (Jan 10), not injected_at


def test_conversation_max_timestamp_considers_probes_too():
    records = [
        _record(memory_id="m1", conversation_id="c1", injected_at="2026-01-01T00:00:00",
                censored=True, invalidated_at=None,
                probes=[{"probe_at": "2026-01-02T00:00:00", "expected_answer_source": "original"}]),
        _record(memory_id="m2", conversation_id="c1", injected_at="2026-01-01T00:00:00",
                censored=True, invalidated_at=None,
                probes=[{"probe_at": "2026-01-20T00:00:00", "expected_answer_source": "original"}]),
    ]
    conv_max = conversation_max_timestamp(records)
    assert conv_max["c1"].isoformat() == "2026-01-20T00:00:00"
    # m1's own resolve_censoring_time uses its own last probe (Jan 2), since
    # it has probes -- conv_max is only the fallback for probe-less records
    _censoring_time, reason = resolve_censoring_time(records[0], conv_max)
    assert reason == "censored_last_probe"


def test_duration_floored_at_min_duration_days_never_zero_or_negative():
    # invalidated_at == injected_at: a naive (t1 - t0) would be exactly 0,
    # which the survival loss cannot accept (must be strictly positive)
    rec = _record(censored=False, invalidated_at="2026-01-01T00:00:00")
    duration, event, _ = compute_target(rec, conv_max={})
    assert duration == MIN_DURATION_DAYS
    assert duration > 0


def test_label_records_populates_all_three_fields_and_reasons_are_valid():
    records = [
        _record(memory_id="m1", censored=False, invalidated_at="2026-01-03T00:00:00"),
        _record(memory_id="m2", conversation_id="c1", censored=True, invalidated_at=None,
                probes=[{"probe_at": "2026-01-02T00:00:00", "expected_answer_source": "original"}]),
    ]
    labeled = label_records(records)
    for r in labeled:
        assert r["duration_days"] > 0
        assert r["event_observed"] in (0, 1)
        assert r["censor_reason"] in VALID_CENSOR_REASONS


def test_schema_invariant_censored_iff_invalidated_at_is_none():
    # the invariant every record in the dataset must satisfy -- an observed
    # event without invalidated_at, or a censored record WITH one, is a bug
    good = MemoryRecord.from_dict(_record(censored=False, invalidated_at="2026-01-03T00:00:00"))
    good.validate()  # should not raise

    bad = MemoryRecord.from_dict(_record(censored=True, invalidated_at="2026-01-03T00:00:00"))
    try:
        bad.validate()
        assert False, "expected AssertionError for censored=True with invalidated_at set"
    except AssertionError:
        pass
