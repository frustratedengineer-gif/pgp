"""Periodic reflection (the "periodic importance decay" box): as a memory
ages relative to its OWN predicted TTL, its importance and utility scores
decay -- a memory a day past its predicted expiry is worth ranking lower in
retrieval even before forgetting.py actually removes it (forgetting has its
own, separate threshold/policy; this is a softer, continuous signal for
retrieval/scoring.py in between reflection sweeps).

Does NOT re-run the joint model or feature extractors (that would need the
memory's raw text + re-encoding, a heavier operation better suited to a
less frequent "full refresh" pass, not implemented here) -- this is the
cheap, frequent half of reflection: decay existing scores based on age
alone. A real deployment would run this on every retrieval or on a fast
timer, and reserve a full-model refresh for occasional batches.
"""
from datetime import datetime

from .audit import AuditLog
from .store.base import MemoryStore

DEFAULT_DECAY_RATE = 0.5  # importance/utility multiplied by this per multiple of predicted_ttl_days elapsed


def apply_decay(store: MemoryStore, audit_log: AuditLog, as_of: datetime | None = None,
                 decay_rate: float = DEFAULT_DECAY_RATE) -> int:
    """Returns the number of memories decayed this pass."""
    n_decayed = 0
    for obj in store.all(include_forgotten=False):
        ttl = max(obj.predicted_ttl_days, 1e-6)
        overdue_ratio = max(obj.age_days(as_of) / ttl - 1.0, 0.0)
        if overdue_ratio <= 0:
            continue

        factor = decay_rate ** overdue_ratio
        old_importance, old_utility = obj.importance, obj.utility_prob
        obj.importance = max(obj.importance * factor, 0.0)
        obj.utility_prob = max(obj.utility_prob * factor, 0.0)
        audit_log.log("updated", obj.memory_id, "reflection_decay",
                       extra={"overdue_ratio": round(overdue_ratio, 3),
                              "importance_before": round(old_importance, 4), "importance_after": round(obj.importance, 4),
                              "utility_before": round(old_utility, 4), "utility_after": round(obj.utility_prob, 4)})
        n_decayed += 1
    return n_decayed
