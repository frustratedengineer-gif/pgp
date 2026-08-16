"""
Downstream memory-policy baseline: store everything, forget nothing. The
upper bound on both storage (grows unboundedly with conversation length)
and, presumably, QA accuracy (nothing useful is ever evicted) -- the
policy comparison in `scripts/run_downstream_qa_eval.py` is really asking
"how much of no_forget's accuracy can a bounded-storage policy keep?", and
this is the reference ceiling for that question.
"""
from memorylife.memory.audit import AuditLog
from memorylife.memory.store.base import MemoryStore


def sweep(store: MemoryStore, audit_log: AuditLog, **kwargs) -> list[str]:
    return []
