"""Reranks a candidate pool (from MemoryStore.search's raw cosine
similarity order) by the combined score in scoring.py, and returns the top
k. Kept separate from retriever.py so a future cross-encoder/LLM reranker
can be swapped in here without touching the retrieval orchestration.
"""
from ..memory.memory_object import MemoryObject
from .scoring import ScoringWeights, combined_score


def rerank(candidates: list[tuple[MemoryObject, float]], k: int,
           weights: ScoringWeights = ScoringWeights()) -> list[tuple[MemoryObject, float]]:
    """candidates: [(MemoryObject, similarity), ...], any order.
    Returns the top k by combined_score, as [(MemoryObject, combined_score), ...]."""
    scored = [(obj, combined_score(sim, obj, weights)) for obj, sim in candidates]
    scored.sort(key=lambda pair: -pair[1])
    return scored[:k]
