"""
Right-censoring-time resolution.

Convention (see docs/benchmark_card.md for the full write-up):
  - censored=False -> the event was observed; the invalidated_at timestamp
    on the record IS the event time. No work needed here.
  - censored=True, has probes (synthetic records) -> censor at the last
    probe_at: the last time we actively checked and the memory had not
    been invalidated yet.
  - censored=True, no probes (real: longmemeval/locomo) -> administrative
    censoring at the latest timestamp seen anywhere in the same
    conversation_id (injected_at / invalidated_at / probe_at of every
    record sharing that conversation). This is the last point at which we
    observed that conversation at all.

Durations are floored at MIN_DURATION_DAYS to stay strictly positive for
the survival loss.
"""
from collections import defaultdict
from datetime import datetime

MIN_DURATION_DAYS = 0.01


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def conversation_max_timestamp(records: list[dict]) -> dict[str, datetime]:
    """conversation_id -> latest timestamp seen anywhere in that conversation."""
    conv_max: dict[str, datetime] = {}
    for d in records:
        ts_list = [_parse(d["injected_at"])]
        if d.get("invalidated_at"):
            ts_list.append(_parse(d["invalidated_at"]))
        for p in d.get("probes") or []:
            ts_list.append(_parse(p["probe_at"]))
        cid = d["conversation_id"]
        local_max = max(ts_list)
        if cid not in conv_max or local_max > conv_max[cid]:
            conv_max[cid] = local_max
    return conv_max


def resolve_censoring_time(record: dict, conv_max: dict[str, datetime]) -> tuple[datetime, str]:
    """Returns (censoring_time, reason) for a record with censored=True."""
    probes = record.get("probes") or []
    if probes:
        return max(_parse(p["probe_at"]) for p in probes), "censored_last_probe"
    return conv_max[record["conversation_id"]], "censored_conversation_end"
