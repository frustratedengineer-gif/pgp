"""Abstract interface every memory-store backend implements."""
from abc import ABC, abstractmethod

import numpy as np

from ..memory_object import MemoryObject


class MemoryStore(ABC):
    @abstractmethod
    def add(self, obj: MemoryObject) -> None:
        ...

    @abstractmethod
    def get(self, memory_id: str) -> MemoryObject | None:
        ...

    @abstractmethod
    def all(self, include_forgotten: bool = False) -> list[MemoryObject]:
        ...

    @abstractmethod
    def remove(self, memory_id: str) -> None:
        """Hard delete. Prefer memory.forgetting's soft-delete (status =
        STATUS_FORGOTTEN) so the audit log can still explain what happened;
        this is for genuine cleanup, e.g. after compaction merges records."""
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10,
               include_forgotten: bool = False) -> list[tuple[MemoryObject, float]]:
        """Returns up to k (MemoryObject, cosine_similarity) pairs, sorted
        by similarity descending."""
        ...
