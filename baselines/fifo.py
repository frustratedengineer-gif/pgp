"""
Downstream memory-policy baseline: forget the OLDEST memories (by
`created_at`) once the active store exceeds `capacity`, regardless of
whether they're still true/useful. The naive baseline every real memory
system implicitly argues it's better than -- see
`scripts/run_downstream_qa_eval.py` for the actual comparison.
"""
from memorylife.memory.audit import AuditLog
from memorylife.memory.memory_object import STATUS_FORGOTTEN
from memorylife.memory.store.base import MemoryStore


def sweep(store: MemoryStore, audit_log: AuditLog, capacity: int, **kwargs) -> list[str]:
    active = store.all(include_forgotten=False)
    n_over = len(active) - capacity
    if n_over <= 0:
        return []

    active_sorted = sorted(active, key=lambda o: o.created_at)  # oldest first
    to_forget = active_sorted[:n_over]
    forgotten_ids = []
    for obj in to_forget:
        obj.status = STATUS_FORGOTTEN
        audit_log.log("forgotten", obj.memory_id, "fifo_capacity_evicted",
                       extra={"capacity": capacity, "created_at": obj.created_at})
        forgotten_ids.append(obj.memory_id)
    return forgotten_ids
