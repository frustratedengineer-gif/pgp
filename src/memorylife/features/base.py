"""Common interface every feature extractor implements (the "Semantic
Feature Extractors" box in the architecture figure -- off-the-shelf
pretrained models per the figure's colour legend, not things this project
trains from scratch).

extract() is split-level (not one-record-at-a-time) because novelty needs
the whole conversation's prior records for context, and batching the HF
pipeline-backed extractors (intent/entities/emotion/contradiction) is much
faster than calling them per-record.
"""
from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractor(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        """records: the split's raw record dicts, in the SAME order as
        embeddings. embeddings: (len(records), encoder_dim) float32 array
        from the sentence encoder, needed only by extractors that compare a
        record against other records (e.g. novelty, contradiction) --
        others ignore it.

        Returns (len(records), self.dim) float32."""
        ...
