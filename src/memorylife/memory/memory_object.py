"""
Memory Object (the "Memory Object" box): the unit the memory store holds.
Bundles a memory's text/embedding with everything the joint model predicts
about its lifecycle -- built from JointLifecyclePredictor's outputs (Week
5) plus heads/importance.py's heuristic (see that module's docstring for
why importance is heuristic, not predicted by the joint model).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..heads.action import ACTION_LABELS

STATUS_ACTIVE = "active"
STATUS_FORGOTTEN = "forgotten"


@dataclass
class MemoryObject:
    memory_id: str
    text: str
    embedding: np.ndarray

    importance: float  # heads/importance.py heuristic, in [0, 1]
    predicted_ttl_days: float  # from the Lifetime/survival head: E[S(t|z) > 0], see forgetting.py
    action: str  # one of ACTION_LABELS, from the Action head
    utility_prob: float  # P(retrieved again), from the Future-utility head

    conversation_id: str
    source: str
    category: str
    created_at: str  # ISO-8601, == injected_at from the source record

    status: str = STATUS_ACTIVE
    provenance: dict = field(default_factory=dict)  # original record fields kept for audit/debugging

    def __post_init__(self):
        assert self.action in ACTION_LABELS, f"bad action: {self.action}"
        assert 0.0 <= self.importance <= 1.0, f"importance out of [0,1]: {self.importance}"
        assert 0.0 <= self.utility_prob <= 1.0, f"utility_prob out of [0,1]: {self.utility_prob}"

    def age_days(self, as_of: datetime | None = None) -> float:
        """as_of: the reference "now" to measure age against. Defaults to
        real wall-clock time, but callers replaying dataset conversations
        (all timestamped in the dataset's own timeline, not necessarily
        close to real wall-clock "now") should pass the conversation's own
        current timestamp explicitly -- see inference/pipeline.py."""
        created = datetime.fromisoformat(self.created_at)
        if as_of is None:
            as_of = datetime.now(timezone.utc) if created.tzinfo else datetime.now()
        return (as_of - created).total_seconds() / 86400.0

    def is_expired(self, as_of: datetime | None = None) -> bool:
        return self.age_days(as_of) > self.predicted_ttl_days
