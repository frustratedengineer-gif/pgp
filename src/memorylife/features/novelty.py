"""Novelty: how different is this memory from what's already been said
earlier in the same conversation? Retrieval-conditioned (compares against
the encoder's own embedding space, no new model), and causal: only
memories that occurred STRICTLY BEFORE this one (by injected_at) within the
same conversation_id are considered "already known" -- comparing against
later memories would leak future information a real deployed system
wouldn't have at write time.
"""
from datetime import datetime

import numpy as np

from .base import FeatureExtractor


class NoveltyFeatures(FeatureExtractor):
    """1 feature: novelty = 1 - max_cosine_similarity to any earlier memory
    in the same conversation (embeddings are already L2-normalized by the
    BGE encoder, so cosine similarity == dot product). A record with no
    earlier memories in its conversation gets novelty=1.0 (maximally
    novel -- nothing to compare against)."""

    @property
    def dim(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "novelty"

    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        if embeddings is None:
            raise ValueError("NoveltyFeatures requires embeddings (compares against prior memories' embeddings)")

        by_conv: dict[str, list[int]] = {}
        for idx, r in enumerate(records):
            by_conv.setdefault(r["conversation_id"], []).append(idx)

        out = np.ones((len(records), 1), dtype=np.float32)
        for conv_id, idxs in by_conv.items():
            idxs_sorted = sorted(idxs, key=lambda i: datetime.fromisoformat(records[i]["injected_at"]))
            seen_embeddings = []
            for i in idxs_sorted:
                if seen_embeddings:
                    sims = embeddings[i] @ np.stack(seen_embeddings).T
                    out[i, 0] = 1.0 - float(sims.max())
                seen_embeddings.append(embeddings[i])
        return out
