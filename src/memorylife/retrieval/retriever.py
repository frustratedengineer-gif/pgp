"""The "Retriever" box: given a query, pulls a similarity-based candidate
pool from the memory store, then reranks it by sim + importance + utility
(scoring.py). Excludes forgotten memories by construction (MemoryStore's
default `include_forgotten=False` on both search and all).
"""
from ..memory.store.base import MemoryStore
from .index import QueryEncoder
from .rerank import rerank
from .scoring import ScoringWeights

DEFAULT_CANDIDATE_POOL_MULTIPLIER = 5  # pull 5x the requested k before reranking, so scoring can reorder meaningfully


class Retriever:
    def __init__(self, store: MemoryStore, query_encoder: QueryEncoder, weights: ScoringWeights = ScoringWeights()):
        self.store = store
        self.query_encoder = query_encoder
        self.weights = weights

    def retrieve(self, query: str, k: int = 5):
        query_embedding = self.query_encoder.encode(query)
        candidate_pool = self.store.search(query_embedding, k=k * DEFAULT_CANDIDATE_POOL_MULTIPLIER)
        return rerank(candidate_pool, k=k, weights=self.weights)
