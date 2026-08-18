"""
Downstream QA metrics: EM (exact match) and token-F1, the standard SQuAD-
style scoring for free-form QA against a reference answer. Cheap and
deterministic -- used for the bulk of `scripts/run_downstream_qa_eval.py`'s
scoring, since LLM-judge scoring (`llm_judge_score` below, using
`inference/prompts/judge.txt`) costs a second LLM call per QA pair and is
reserved for a smaller subsample by default (see that script's
`--judge-sample-frac`).

Normalization matches the original SQuAD eval script's convention (lower,
strip punctuation/articles, collapse whitespace) so scores are comparable
to how LoCoMo/LongMemEval's own papers report EM/F1, not a bespoke metric.
"""
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_qa(prediction: str, reference: str) -> dict:
    return {"em": exact_match(prediction, reference), "f1": token_f1(prediction, reference)}


REFUSAL_RE = re.compile(
    r"don'?t have (that |this |any |the )?information|"
    r"do not have (that |this |any |the )?information|"
    r"no information (is |was )?available|"
    r"don'?t know|do not know|"
    r"not (be )?able to (determine|answer|find|tell)|"
    r"unable to (determine|answer|find|tell)|"
    r"no relevant information",
    re.IGNORECASE,
)


def is_refusal(prediction: str) -> bool:
    """Detects whether a QA answer refuses to answer (matches the exact
    behavior scripts/inference/prompts/qa_grounded.txt instructs: "say you
    don't have that information yet"), used to score refusal precision/
    recall on LoCoMo's category-5 adversarial questions
    (scripts/eval_refusal.py) -- a real system should refuse these, not
    hallucinate an answer against the adversarial decoy. Matches ANY
    refusal signal in the text, including a partial refusal embedded in an
    otherwise-answered compound question (e.g. "$15 on car wash. I don't
    have information on a parking ticket."), not just a bare refusal
    sentence -- calibrated against real GPT-4o outputs collected in Week 6,
    not written blind."""
    return bool(REFUSAL_RE.search(prediction))


JUDGE_VERDICT_RE = re.compile(r"\b(correct|incorrect)\b", re.IGNORECASE)


def llm_judge_score(question: str, prediction: str, reference: str, model: str = "openai/gpt-4o") -> tuple[float, dict]:
    """(score, usage): score is 1.0 if the judge says the prediction is
    substantively correct given the reference answer, 0.0 otherwise; usage
    is the API call's token-usage dict, so callers scoring many predictions
    can track real spend (see scripts/judge_downstream_qa.py) rather than
    just estimating it. Needed because EM/F1 penalize correct-but-
    differently-worded answers (e.g. "Business Administration" vs "a
    degree in Business Administration") -- a real limitation of the cheap
    metrics above, not a flaw specific to our system, but one that matters
    for a fair downstream comparison. See prompts/judge.txt."""
    from pathlib import Path

    from ..inference.llm_client import chat_completion

    template = (Path(__file__).parent.parent / "inference" / "prompts" / "judge.txt").read_text()
    prompt = template.format(question=question, prediction=prediction, reference=reference)
    text, usage = chat_completion([{"role": "user", "content": prompt}], model=model, max_tokens=10)
    match = JUDGE_VERDICT_RE.search(text)
    score = 1.0 if match and match.group(1).lower() == "correct" else 0.0
    return score, usage
