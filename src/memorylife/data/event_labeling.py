"""
Defines T (duration_days) and delta (event_observed) for every record --
the (T, delta) reference events the survival model and all baselines are
trained/evaluated against.
"""
from datetime import datetime

from .censoring import MIN_DURATION_DAYS, conversation_max_timestamp, resolve_censoring_time


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def compute_target(record: dict, conv_max: dict[str, datetime]) -> tuple[float, int, str]:
    """Returns (duration_days, event_observed, censor_reason) for one record."""
    t0 = _parse(record["injected_at"])

    if not record["censored"]:
        t1 = _parse(record["invalidated_at"])
        event, reason = 1, "observed_event"
    else:
        t1, reason = resolve_censoring_time(record, conv_max)
        event = 0

    duration_days = max((t1 - t0).total_seconds() / 86400.0, MIN_DURATION_DAYS)
    return duration_days, event, reason


def label_records(records: list[dict]) -> list[dict]:
    """Adds duration_days / event_observed / censor_reason to every record
    (in place, and returned) using conversation-level censoring times
    computed over this same batch of records."""
    conv_max = conversation_max_timestamp(records)
    for d in records:
        duration_days, event, reason = compute_target(d, conv_max)
        d["duration_days"] = round(duration_days, 4)
        d["event_observed"] = event
        d["censor_reason"] = reason
    return records
