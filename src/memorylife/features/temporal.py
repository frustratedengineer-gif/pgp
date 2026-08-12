"""Temporal cues (dates, deadlines, tense, explicit expiry hints) --
deterministic regex, no model, no training. Complements the learned
Lifetime head with an interpretable signal a reviewer can sanity-check by
eye (e.g. "why did the model think this was short-lived?" -> "it matched
the deadline pattern").
"""
import re

import numpy as np

from .base import FeatureExtractor

EXPLICIT_DATE_PAT = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?|january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\b", re.IGNORECASE,
)
DEADLINE_PAT = re.compile(
    r"\b(deadline|due (on|date|by)|expir\w*|appointment|meeting|flight|reservation)\b",
    re.IGNORECASE,
)
NEAR_TERM_PAT = re.compile(
    r"\b(today|tonight|tomorrow|this morning|this afternoon|this evening|"
    r"this week|next week)\b", re.IGNORECASE,
)
PERMANENCE_PAT = re.compile(
    r"\b(always|forever|permanently|never|my name is|i was born|i live in|"
    r"i am \d|i'm \d)\b", re.IGNORECASE,
)
PAST_TENSE_PAT = re.compile(r"\b(\w+ed|was|were|had|used to)\b", re.IGNORECASE)
FUTURE_TENSE_PAT = re.compile(r"\b(will|going to|plan to|about to|shall)\b", re.IGNORECASE)


class TemporalFeatures(FeatureExtractor):
    """6 features: [has_explicit_date, has_deadline_word, has_near_term_word,
    has_permanence_word, has_past_tense, has_future_tense], each in {0.,1.}."""

    FEATURE_NAMES = (
        "has_explicit_date", "has_deadline_word", "has_near_term_word",
        "has_permanence_word", "has_past_tense", "has_future_tense",
    )

    @property
    def dim(self) -> int:
        return len(self.FEATURE_NAMES)

    @property
    def name(self) -> str:
        return "temporal"

    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros((len(records), self.dim), dtype=np.float32)
        for i, r in enumerate(records):
            text = r["text"]
            out[i, 0] = float(bool(EXPLICIT_DATE_PAT.search(text)))
            out[i, 1] = float(bool(DEADLINE_PAT.search(text)))
            out[i, 2] = float(bool(NEAR_TERM_PAT.search(text)))
            out[i, 3] = float(bool(PERMANENCE_PAT.search(text)))
            out[i, 4] = float(bool(PAST_TENSE_PAT.search(text)))
            out[i, 5] = float(bool(FUTURE_TENSE_PAT.search(text)))
        return out
