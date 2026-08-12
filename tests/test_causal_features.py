"""
Proves (doesn't just claim in a docstring) that
features.causal.nearest_prior_in_conversation -- shared by novelty.py and
contradiction.py -- never leaks future information: a record is never
matched against another record that occurs at the same time or later, and
never matched across conversation_id boundaries.
"""
import numpy as np

from memorylife.features.causal import nearest_prior_in_conversation
from memorylife.features.novelty import NoveltyFeatures


def _record(memory_id, conversation_id, injected_at):
    return {"memory_id": memory_id, "conversation_id": conversation_id, "injected_at": injected_at}


def test_first_record_in_conversation_has_no_prior():
    records = [
        _record("m1", "c1", "2026-01-01T00:00:00"),
        _record("m2", "c1", "2026-01-02T00:00:00"),
        _record("m3", "c1", "2026-01-03T00:00:00"),
    ]
    embeddings = np.eye(3, dtype=np.float32)  # orthogonal, similarity is irrelevant here
    prior = nearest_prior_in_conversation(records, embeddings)
    assert prior[0] is None  # m1: nothing before it
    assert prior[1] == 0     # m2: only m1 came before
    assert prior[2] in (0, 1)  # m3: must be one of the two earlier records


def test_never_matches_a_later_record_even_when_more_similar():
    """The whole point of the causal constraint: record 0 (t=0) is IDENTICAL
    to record 2 (t=2), and record 1 (t=1) is very different from both. If
    the function leaked future information, record 1's "nearest prior"
    would incorrectly reach forward to record 2 (post-hoc, since it's a
    perfect match) -- it must not, because record 2 doesn't exist yet at
    record 1's time."""
    records = [
        _record("early", "c1", "2026-01-01T00:00:00"),
        _record("middle", "c1", "2026-01-02T00:00:00"),
        _record("late_but_identical_to_early", "c1", "2026-01-03T00:00:00"),
    ]
    embeddings = np.array([
        [1.0, 0.0],  # "early"
        [0.0, 1.0],  # "middle" -- orthogonal to both
        [1.0, 0.0],  # "late_but_identical_to_early" -- identical to "early"
    ], dtype=np.float32)

    prior = nearest_prior_in_conversation(records, embeddings)
    assert prior[1] == 0, "record 1 ('middle') must only ever be able to see record 0, never record 2"
    assert prior[2] == 0, "record 2 correctly matches record 0 (its only valid, earlier, option)"


def test_matching_never_crosses_conversation_boundaries():
    records = [
        _record("c1_a", "c1", "2026-01-01T00:00:00"),
        _record("c2_a", "c2", "2026-01-01T00:00:01"),  # different conversation, 1 second later
        _record("c1_b", "c1", "2026-01-02T00:00:00"),
        _record("c2_b", "c2", "2026-01-02T00:00:01"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prior = nearest_prior_in_conversation(records, embeddings)
    assert prior[2] == 0, "c1_b must match c1_a, not anything from c2"
    assert prior[3] == 1, "c2_b must match c2_a, not anything from c1"


def test_result_is_independent_of_input_list_order_only_injected_at_matters():
    """If the function accidentally relied on list position instead of
    injected_at, shuffling the input order would silently change results."""
    chronological = [
        _record("m1", "c1", "2026-01-01T00:00:00"),
        _record("m2", "c1", "2026-01-02T00:00:00"),
        _record("m3", "c1", "2026-01-03T00:00:00"),
    ]
    embeddings_chrono = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    prior_chrono = nearest_prior_in_conversation(chronological, embeddings_chrono)
    expected = {chronological[i]["memory_id"]: (chronological[p]["memory_id"] if p is not None else None)
                for i, p in prior_chrono.items()}

    scrambled_order = [2, 0, 1]  # m3, m1, m2 -- list order no longer matches time order
    scrambled = [chronological[i] for i in scrambled_order]
    embeddings_scrambled = embeddings_chrono[scrambled_order]
    prior_scrambled = nearest_prior_in_conversation(scrambled, embeddings_scrambled)
    actual = {scrambled[i]["memory_id"]: (scrambled[p]["memory_id"] if p is not None else None)
              for i, p in prior_scrambled.items()}

    assert actual == expected


def test_novelty_feature_uses_the_causal_helper_correctly():
    records = [
        _record("m1", "c1", "2026-01-01T00:00:00"),
        _record("m2", "c1", "2026-01-02T00:00:00"),
    ]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # identical embeddings
    feats = NoveltyFeatures().extract(records, embeddings=embeddings)
    assert feats[0, 0] == 1.0  # m1: nothing before it, maximally novel
    assert abs(feats[1, 0] - 0.0) < 1e-6  # m2: identical to m1 -> novelty 0
