"""final_active_ids and remaining_life_fraction (scripts/run_downstream_qa_eval.py)
decide exactly which memories survive under each policy -- every Week-6
downstream QA/evidence-retention number is only as trustworthy as this
set logic. `ours_utility`/`ours_combo` (the Fix #2 policies) are tested
directly against `fifo`/`lru`/`ours`, since the whole Week-6 argument is
a comparison between them at the SAME capacity."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from memorylife.memory.memory_object import MemoryObject

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_downstream_qa_eval import final_active_ids, remaining_life_fraction  # noqa: E402

AS_OF = datetime(2026, 1, 10)


def _obj(mid, created_at, ttl_days, action="store", utility_prob=0.5):
    return MemoryObject(
        memory_id=mid, text=f"text-{mid}", embedding=np.zeros(4, dtype="float32"),
        importance=0.5, predicted_ttl_days=ttl_days, action=action, utility_prob=utility_prob,
        conversation_id="c1", source="test", category="test", created_at=created_at,
    )


# three memories, evenly spaced, all created before AS_OF
OBJECTS = [
    _obj("m1", "2026-01-01T00:00:00", ttl_days=3, action="store", utility_prob=0.1),  # age=9, expired
    _obj("m2", "2026-01-05T00:00:00", ttl_days=30, action="forget", utility_prob=0.9),  # not expired, but forget
    _obj("m3", "2026-01-08T00:00:00", ttl_days=30, action="store", utility_prob=0.5),  # not expired, kept
]
LAST_REFERENCED = {"m1": datetime(2026, 1, 9).timestamp(), "m3": datetime(2026, 1, 2).timestamp()}


def test_no_forget_keeps_everything_regardless_of_capacity():
    ids = final_active_ids(OBJECTS, "no_forget", capacity=1, last_referenced={}, as_of=AS_OF)
    assert ids == {"m1", "m2", "m3"}


def test_ours_drops_expired_and_forget_flagged_memories():
    ids = final_active_ids(OBJECTS, "ours", capacity=None, last_referenced={}, as_of=AS_OF)
    assert ids == {"m3"}  # m1 expired (age 9 > ttl 3), m2 flagged "forget"


def test_fifo_keeps_the_n_most_recently_created():
    ids = final_active_ids(OBJECTS, "fifo", capacity=2, last_referenced={}, as_of=AS_OF)
    assert ids == {"m2", "m3"}  # created 01-05 and 01-08, drops m1 (01-01)


def test_lru_keeps_the_n_most_recently_referenced():
    # m1 referenced 01-09 (most recent), m3 referenced 01-02, m2 never referenced (-inf)
    ids = final_active_ids(OBJECTS, "lru", capacity=2, last_referenced=LAST_REFERENCED, as_of=AS_OF)
    assert ids == {"m1", "m3"}


def test_ours_utility_ranks_by_utility_prob_descending():
    ids = final_active_ids(OBJECTS, "ours_utility", capacity=2, last_referenced={}, as_of=AS_OF)
    assert ids == {"m2", "m3"}  # utility_prob 0.9, 0.5 beat m1's 0.1


def test_ours_combo_blends_utility_and_remaining_life():
    ids = final_active_ids(OBJECTS, "ours_combo", capacity=1, last_referenced={}, as_of=AS_OF)
    # m1: utility=0.1, remaining_life=0 (expired) -> score 0.05
    # m2: utility=0.9, remaining_life=1-4/30 (age 5 days) -> score high
    # m3: utility=0.5, remaining_life=1-2/30 -> score mid
    assert ids == {"m2"}


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        final_active_ids(OBJECTS, "not_a_real_policy", capacity=1, last_referenced={}, as_of=AS_OF)


def test_remaining_life_fraction_is_1_for_a_brand_new_memory():
    obj = _obj("fresh", AS_OF.isoformat(), ttl_days=10)
    assert remaining_life_fraction(obj, AS_OF) == 1.0


def test_remaining_life_fraction_is_0_once_past_ttl():
    obj = _obj("old", "2025-12-01T00:00:00", ttl_days=5)
    assert remaining_life_fraction(obj, AS_OF) == 0.0


def test_remaining_life_fraction_is_0_for_a_non_positive_ttl():
    obj = _obj("zero_ttl", "2026-01-09T00:00:00", ttl_days=0)
    assert remaining_life_fraction(obj, AS_OF) == 0.0
