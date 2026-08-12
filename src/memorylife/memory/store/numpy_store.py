"""In-memory, brute-force-cosine-search MemoryStore. The default backend:
at MemoryLifeBench's scale (~10K memories) a numpy matmul against every
embedding is a few milliseconds, so there's no need for FAISS/Chroma's
approximate-search machinery yet -- faiss_store.py/chroma_store.py stay
documented stubs for when the memory count grows past what brute force
handles comfortably (see docs/reproducibility.md's known-gaps section).
"""
import numpy as np

from ..memory_object import STATUS_ACTIVE, MemoryObject
from .base import MemoryStore


class NumpyMemoryStore(MemoryStore):
    def __init__(self):
        self._objects: dict[str, MemoryObject] = {}

    def add(self, obj: MemoryObject) -> None:
        self._objects[obj.memory_id] = obj

    def get(self, memory_id: str) -> MemoryObject | None:
        return self._objects.get(memory_id)

    def all(self, include_forgotten: bool = False) -> list[MemoryObject]:
        if include_forgotten:
            return list(self._objects.values())
        return [o for o in self._objects.values() if o.status == STATUS_ACTIVE]

    def remove(self, memory_id: str) -> None:
        self._objects.pop(memory_id, None)

    def search(self, query_embedding: np.ndarray, k: int = 10,
               include_forgotten: bool = False) -> list[tuple[MemoryObject, float]]:
        candidates = self.all(include_forgotten=include_forgotten)
        if not candidates:
            return []
        emb_matrix = np.stack([c.embedding for c in candidates])
        # embeddings are L2-normalized (BGEEncoder), so dot product == cosine similarity
        sims = emb_matrix @ query_embedding
        top_k_idx = np.argsort(-sims)[:k]
        return [(candidates[i], float(sims[i])) for i in top_k_idx]
