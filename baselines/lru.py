"""
Downstream memory-policy baseline: forget the LEAST-RECENTLY-RETRIEVED
memories once the active store exceeds `capacity`. Needs access-time
tracking that isn't part of `MemoryObject`'s core schema on purpose (it's a
baseline-specific concept, not something our own system needs) -- the
caller (`scripts/run_downstream_qa_eval.py`) maintains
`last_retrieved_at: dict[memory_id, float]` externally and updates it after
every retrieval, passing it in here.
"""
from memorylife.memory.audit import AuditLog
from memorylife.memory.memory_object import STATUS_FORGOTTEN
from memorylife.memory.store.base import MemoryStore


def sweep(store: MemoryStore, audit_log: AuditLog, capacity: int,
          last_retrieved_at: dict[str, float], **kwargs) -> list[str]:
    active = store.all(include_forgotten=False)
    n_over = len(active) - capacity
    if n_over <= 0:
        return []

    # never-retrieved memories are treated as retrieved at time -inf --
    # they're evicted first, same intuition as a real LRU cache's cold entries
    active_sorted = sorted(active, key=lambda o: last_retrieved_at.get(o.memory_id, float("-inf")))
    to_forget = active_sorted[:n_over]
    forgotten_ids = []
    for obj in to_forget:
        obj.status = STATUS_FORGOTTEN
        audit_log.log("forgotten", obj.memory_id, "lru_capacity_evicted",
                       extra={"capacity": capacity,
                              "last_retrieved_at": last_retrieved_at.get(obj.memory_id)})
        forgotten_ids.append(obj.memory_id)
    return forgotten_ids
