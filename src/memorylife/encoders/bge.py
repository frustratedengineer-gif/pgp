"""BGE sentence encoder (the "Sentence Encoder" box in the architecture figure).

BAAI/bge-base-en-v1.5, 768-d, matches the fused-vector width used downstream.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import SentenceEncoder

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"


class BGEEncoder(SentenceEncoder):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cuda"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        # BGE's query-instruction prefix is only for queries, not stored
        # passages, so memory text is encoded as-is.
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)
