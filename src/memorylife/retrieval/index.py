"""Query encoding: turns an incoming query string into the same embedding
space the memory store's vectors live in. A thin wrapper, not a duplicate
of memory/store/numpy_store.py's search index -- that module holds the
STORED memory vectors; this one is specifically for embedding the QUERY
side of retrieval, reusing the identical encoder (same model, same
normalization) so cosine similarity is meaningful.
"""
import numpy as np

from ..encoders.bge import BGEEncoder


class QueryEncoder:
    def __init__(self, encoder: BGEEncoder | None = None, device: str = "cpu"):
        self.encoder = encoder or BGEEncoder(device=device)

    def encode(self, query: str) -> np.ndarray:
        return self.encoder.encode([query])[0]
