"""Named-entity counts (the "Entities (NER)" box) -- off-the-shelf
dslim/bert-base-NER, no training. A memory naming a specific PERSON/ORG/LOC
tends to be a more concrete, checkable fact than a vague statement, which
is exactly the kind of auxiliary signal the architecture figure calls for
fusing alongside the raw embedding.
"""
import numpy as np

from .base import FeatureExtractor

DEFAULT_MODEL_NAME = "dslim/bert-base-NER"
ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")  # dslim/bert-base-NER's label set


class EntityFeatures(FeatureExtractor):
    """5 features: [count_PER, count_ORG, count_LOC, count_MISC, total_entities]."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        from transformers import pipeline
        # aggregation_strategy="simple" merges wordpiece sub-tokens into whole entities
        self._pipe = pipeline("ner", model=model_name, aggregation_strategy="simple",
                               device=0 if device == "cuda" else -1)

    @property
    def dim(self) -> int:
        return len(ENTITY_TYPES) + 1

    @property
    def name(self) -> str:
        return "entities"

    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        texts = [r["text"][:500] for r in records]
        out = np.zeros((len(records), self.dim), dtype=np.float32)
        results = self._pipe(texts, batch_size=64)
        for i, ents in enumerate(results):
            for e in ents:
                etype = e.get("entity_group", "MISC")
                if etype in ENTITY_TYPES:
                    out[i, ENTITY_TYPES.index(etype)] += 1.0
            out[i, -1] = float(len(ents))
        return out
