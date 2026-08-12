"""Novelty: how different is this memory from what's already been said
earlier in the same conversation? Retrieval-conditioned (compares against
the encoder's own embedding space, no new model), and causal: only
memories that occurred STRICTLY BEFORE this one (by injected_at) within the
same conversation_id are considered "already known" -- comparing against
later memories would leak future information a real deployed system
wouldn't have at write time.
"""
import numpy as np

from .base import FeatureExtractor
from .causal import nearest_prior_in_conversation


class NoveltyFeatures(FeatureExtractor):
    """1 feature: novelty = 1 - cosine_similarity to the nearest earlier
    memory in the same conversation (embeddings are already L2-normalized
    by the BGE encoder, so cosine similarity == dot product). A record with
    no earlier memories in its conversation gets novelty=1.0 (maximally
    novel -- nothing to compare against). Causal ordering (never compares
    against a later record) is implemented once, in causal.py, and proven
    by tests/test_causal_features.py rather than re-implemented here."""

    @property
    def dim(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "novelty"

    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        if embeddings is None:
            raise ValueError("NoveltyFeatures requires embeddings (compares against prior memories' embeddings)")

        prior = nearest_prior_in_conversation(records, embeddings)
        out = np.ones((len(records), 1), dtype=np.float32)
        for i, prior_i in prior.items():
            if prior_i is not None:
                sim = float(embeddings[i] @ embeddings[prior_i])
                out[i, 0] = 1.0 - sim
        return out
