"""Self-compaction (the "self-compaction: merge redundant -> summary" box):
finds near-duplicate ACTIVE memories within the same conversation and
merges them, keeping the higher-importance one and forgetting the rest with
an audit trail explaining exactly which memory absorbed which.

Not a real summarization step (no LLM call to produce a merged summary
text) -- that's future work; this only decides WHICH memories are
redundant enough to merge and records the decision. Wiring an actual
LLM-generated merge summary in is a small addition on top once this
decision logic is validated (reuse baselines/_openrouter_client.py's
pattern for the LLM call).
"""
from collections import defaultdict

import numpy as np

from .audit import AuditLog
from .memory_object import STATUS_FORGOTTEN, MemoryObject
from .store.base import MemoryStore

DEFAULT_SIMILARITY_THRESHOLD = 0.95


def find_and_merge_duplicates(store: MemoryStore, audit_log: AuditLog,
                               similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> list[tuple[str, str]]:
    """Returns [(kept_id, forgotten_id), ...] pairs actually merged."""
    by_conv: dict[str, list[MemoryObject]] = defaultdict(list)
    for obj in store.all(include_forgotten=False):
        by_conv[obj.conversation_id].append(obj)

    merged_pairs = []
    for conv_id, objs in by_conv.items():
        if len(objs) < 2:
            continue
        emb = np.stack([o.embedding for o in objs])
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1.0)  # never "merge" a memory with itself

        merged_this_conv = set()
        for i, obj_i in enumerate(objs):
            if obj_i.memory_id in merged_this_conv:
                continue
            for j in range(i + 1, len(objs)):
                obj_j = objs[j]
                if obj_j.memory_id in merged_this_conv or sims[i, j] < similarity_threshold:
                    continue
                keep, drop = (obj_i, obj_j) if obj_i.importance >= obj_j.importance else (obj_j, obj_i)
                drop.status = STATUS_FORGOTTEN
                audit_log.log("compacted", drop.memory_id, "merged_near_duplicate",
                               extra={"kept_memory_id": keep.memory_id, "cosine_similarity": float(sims[i, j])})
                merged_this_conv.add(drop.memory_id)
                merged_pairs.append((keep.memory_id, drop.memory_id))

    return merged_pairs
