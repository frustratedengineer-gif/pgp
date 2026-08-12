"""Intent classification (the "Intent" box) -- off-the-shelf zero-shot
classification (no fine-tuned intent classifier exists for this domain, and
training one would need labelled intent data this dataset doesn't have;
zero-shot against a fixed, documented label set is the honest off-the-shelf
substitute the architecture figure's grey/pretrained-model framing calls
for).
"""
import numpy as np

from .base import FeatureExtractor

DEFAULT_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
INTENT_LABELS = (
    "stating a personal fact", "expressing a preference or opinion",
    "describing a plan or event", "asking a question or making a request",
    "giving an instruction", "making small talk",
)


class IntentFeatures(FeatureExtractor):
    """len(INTENT_LABELS) features: zero-shot classification probability
    over the fixed label set above."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        from transformers import pipeline
        self._pipe = pipeline("zero-shot-classification", model=model_name,
                               device=0 if device == "cuda" else -1)

    @property
    def dim(self) -> int:
        return len(INTENT_LABELS)

    @property
    def name(self) -> str:
        return "intent"

    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        texts = [r["text"][:500] for r in records]
        out = np.zeros((len(records), self.dim), dtype=np.float32)
        results = self._pipe(texts, candidate_labels=list(INTENT_LABELS), multi_label=True, batch_size=32)
        if isinstance(results, dict):
            results = [results]
        for i, res in enumerate(results):
            for label, score in zip(res["labels"], res["scores"]):
                out[i, INTENT_LABELS.index(label)] = score
        return out
