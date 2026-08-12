"""Forgetting policy (the "forget + audit log" box): decides which active
memories to soft-delete, driven by the Lifetime head's predicted TTL and
the Action head's explicit "forget" prediction -- NOT importance (the
heuristic importance score is a ranking/retrieval signal, see
retrieval/scoring.py, not a deletion trigger; deleting based on an
unvalidated heuristic would be far riskier than deleting based on the
two heads that actually have real supervision, see heads/importance.py).

Soft-delete only (status -> STATUS_FORGOTTEN), never store.remove() --
matches memory_object.py's status field existing specifically so a
forgotten memory can still be inspected/audited rather than vanishing.
"""
from datetime import datetime

from .audit import AuditLog
from .memory_object import STATUS_FORGOTTEN, MemoryObject
from .store.base import MemoryStore


def sweep(store: MemoryStore, audit_log: AuditLog, as_of: datetime | None = None) -> list[str]:
    """Returns the memory_ids forgotten in this sweep."""
    forgotten_ids = []
    for obj in store.all(include_forgotten=False):
        reason = _forget_reason(obj, as_of)
        if reason is not None:
            obj.status = STATUS_FORGOTTEN
            audit_log.log("forgotten", obj.memory_id, reason,
                           extra={"predicted_ttl_days": obj.predicted_ttl_days,
                                  "age_days": round(obj.age_days(as_of), 2), "action": obj.action})
            forgotten_ids.append(obj.memory_id)
    return forgotten_ids


def _forget_reason(obj: MemoryObject, as_of: datetime | None) -> str | None:
    if obj.action == "forget":
        return "action_head_predicted_forget"
    if obj.is_expired(as_of):
        return "ttl_expired"
    return None
