"""
Concordance index (C-index), the standard survival metric used for the
Week-3 comparison table.

Direction convention: every method here produces one scalar "predicted TTL
/ -risk" per record, where a HIGHER score means a LONGER predicted
remaining lifetime -- the same convention lifelines' own
CoxPHFitter.score() uses internally (it passes `-predict_partial_hazard`,
not the raw hazard, to concordance_index).
"""
from lifelines.utils import concordance_index


def score_direction_note() -> str:
    return (
        "Higher score == longer predicted survival. For a Cox/hazard model, "
        "negate the raw risk/partial-hazard output before calling c_index()."
    )


def c_index(durations, scores, events) -> float:
    """durations: true (possibly censored) time-to-event, in days.
    scores: higher == predicted to survive longer.
    events: 1 if the event was observed, 0 if censored."""
    return concordance_index(durations, scores, events)


def c_index_for_split(scores_by_id: dict[str, float], split: dict) -> float:
    """split: as returned by data-loading helpers, with keys
    'ids', 'durations', 'events' aligned by index."""
    scores = [scores_by_id[mid] for mid in split["ids"]]
    return c_index(split["durations"], scores, split["events"])
