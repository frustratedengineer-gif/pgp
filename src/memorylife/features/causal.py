"""Shared causal-ordering helper for novelty.py and contradiction.py: both
need "for each record, find its most similar EARLIER record in the same
conversation" -- pulled out as one pure, independently-testable function
(see tests/test_causal_features.py) instead of two near-identical inline
implementations that could silently drift apart or each carry their own
leakage bug.
"""
from datetime import datetime

import numpy as np


def nearest_prior_in_conversation(records: list[dict], embeddings: np.ndarray) -> dict[int, int | None]:
    """For each record index i, returns the index of the most similar
    record that occurs STRICTLY BEFORE it (by injected_at) within the same
    conversation_id, or None if no such record exists (first record in its
    conversation). Never returns an index from a different conversation_id,
    and never returns an index that occurs at the same time or later --
    both are asserted below, not just claimed in a docstring.

    embeddings must be L2-normalized (cosine similarity == dot product);
    that's the encoder's contract (see encoders/base.py), not re-checked
    here.
    """
    by_conv: dict[str, list[int]] = {}
    for idx, r in enumerate(records):
        by_conv.setdefault(r["conversation_id"], []).append(idx)

    result: dict[int, int | None] = {}
    for conv_id, idxs in by_conv.items():
        idxs_sorted = sorted(idxs, key=lambda i: datetime.fromisoformat(records[i]["injected_at"]))
        seen_idx: list[int] = []
        seen_emb: list[np.ndarray] = []
        for i in idxs_sorted:
            if seen_emb:
                sims = embeddings[i] @ np.stack(seen_emb).T
                result[i] = seen_idx[int(sims.argmax())]
            else:
                result[i] = None
            seen_idx.append(i)
            seen_emb.append(embeddings[i])
    return result
