"""Common interface every sentence encoder wrapper implements."""
from abc import ABC, abstractmethod
import numpy as np


class SentenceEncoder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Returns a (len(texts), self.dim) float32 array of normalized embeddings."""
        ...
