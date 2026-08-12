"""Emotion / preference intensity (the "Emotion / Preference" box) --
off-the-shelf j-hartmann/emotion-english-distilroberta-base, no training.
Emotionally-charged statements ("I love X", "I'm terrified of Y") are a
plausible signal for how much a memory matters to the user, independent of
how long it stays factually accurate.
"""
import numpy as np

from .base import FeatureExtractor

DEFAULT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")


class EmotionFeatures(FeatureExtractor):
    """len(EMOTION_LABELS) features: full probability distribution over emotions."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        from transformers import pipeline
        self._pipe = pipeline("text-classification", model=model_name, top_k=None,
                               device=0 if device == "cuda" else -1)

    @property
    def dim(self) -> int:
        return len(EMOTION_LABELS)

    @property
    def name(self) -> str:
        return "emotion"

    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        texts = [r["text"][:500] for r in records]
        out = np.zeros((len(records), self.dim), dtype=np.float32)
        results = self._pipe(texts, batch_size=64)
        for i, scores in enumerate(results):
            for item in scores:
                label = item["label"].lower()
                if label in EMOTION_LABELS:
                    out[i, EMOTION_LABELS.index(label)] = item["score"]
        return out
