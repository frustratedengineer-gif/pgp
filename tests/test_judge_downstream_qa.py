"""_cache_key (scripts/judge_downstream_qa.py) is what makes the LLM-judge
cache under artifacts/llm_cache/judge/ actually avoid re-paying for a
question it already judged -- a collision or an unstable hash silently
re-bills or, worse, silently returns another question's cached verdict."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from judge_downstream_qa import _cache_key  # noqa: E402


def test_same_inputs_give_the_same_key():
    a = _cache_key("openai/gpt-4o", "Q?", "pred", "ref")
    b = _cache_key("openai/gpt-4o", "Q?", "pred", "ref")
    assert a == b


def test_changing_any_single_field_changes_the_key():
    base = _cache_key("openai/gpt-4o", "Q?", "pred", "ref")
    assert _cache_key("openai/gpt-4o", "Q?", "different pred", "ref") != base
    assert _cache_key("openai/gpt-4o", "Q?", "pred", "different ref") != base
    assert _cache_key("openai/gpt-4o", "different Q?", "pred", "ref") != base
    assert _cache_key("other-model", "Q?", "pred", "ref") != base


def test_key_is_a_stable_length_hex_digest():
    key = _cache_key("openai/gpt-4o", "Q?", "pred", "ref")
    assert len(key) == 64  # sha256 hexdigest
    assert all(c in "0123456789abcdef" for c in key)
