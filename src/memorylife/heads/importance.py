"""
Importance head (the "Importance head: i^ in [0,1]" box).

IMPORTANT CAVEAT, read before using this anywhere near a paper claim: this
is a HAND-WRITTEN HEURISTIC, not a trained/learned head. Unlike the other
three heads (Lifetime, Action, Future-utility), MemoryLifeBench's schema
(data/schema.py) has no field that measures "how important is this memory
to the user" -- no engagement signal, no explicit rating, nothing. Inventing
a proxy label and training a classifier against it would let a reader
mistake "learned to predict our own heuristic" for "learned true
importance," which is worse than being upfront that this box is not
learned yet. If/when a real importance signal becomes available (e.g. user
engagement logs, explicit ratings), replace this with a trained head using
the same pattern as heads/action.py or heads/future_utility.py.

The heuristic combines four feature-extractor signals already computed by
features/pipeline.py (no separate model, no separate inference cost):
  - permanence marker (temporal.py) -- identity/permanent facts matter
  - deadline/near-term marker (temporal.py) -- actionable near-term items
    matter even though they're short-lived (importance != lifetime)
  - named-entity density (entities.py) -- concrete, checkable facts
  - non-neutral emotion mass (emotion.py) -- emotionally charged statements
Weights are fixed constants below, chosen by inspection, not fit to any
label (there is no label to fit them to).
"""
import numpy as np

from ..features.temporal import TemporalFeatures

W_PERMANENCE = 0.35
W_DEADLINE_OR_NEAR_TERM = 0.25
W_ENTITY_DENSITY = 0.20
W_EMOTION_INTENSITY = 0.20
ENTITY_COUNT_CAP = 3.0  # entity counts beyond this don't add more importance


def importance_score(features: np.ndarray, slices: dict[str, tuple[int, int]]) -> np.ndarray:
    """features: (n, total_feature_dim) as produced by
    features.pipeline.compute_features. slices: the matching feature_slices
    dict. Returns (n,) float32 in [0, 1]."""
    t_start, t_end = slices["temporal"]
    temporal = features[:, t_start:t_end]
    permanence = temporal[:, TemporalFeatures.FEATURE_NAMES.index("has_permanence_word")]
    deadline = temporal[:, TemporalFeatures.FEATURE_NAMES.index("has_deadline_word")]
    near_term = temporal[:, TemporalFeatures.FEATURE_NAMES.index("has_near_term_word")]
    deadline_or_near_term = np.maximum(deadline, near_term)

    e_start, e_end = slices["entities"]
    entity_total = features[:, e_end - 1]  # last column of the entities block is total_entities
    entity_density = np.clip(entity_total / ENTITY_COUNT_CAP, 0.0, 1.0)

    em_start, em_end = slices["emotion"]
    emotion = features[:, em_start:em_end]
    from ..features.emotion import EMOTION_LABELS
    neutral_prob = emotion[:, EMOTION_LABELS.index("neutral")]
    emotion_intensity = 1.0 - neutral_prob

    score = (
        W_PERMANENCE * permanence
        + W_DEADLINE_OR_NEAR_TERM * deadline_or_near_term
        + W_ENTITY_DENSITY * entity_density
        + W_EMOTION_INTENSITY * emotion_intensity
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)
