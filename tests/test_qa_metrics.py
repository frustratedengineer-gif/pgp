"""EM/F1 scoring for the downstream QA eval -- SQuAD-style normalization
must actually behave as claimed (case/punctuation/article-insensitive)."""
from memorylife.evaluation.qa_metrics import exact_match, normalize_answer, token_f1


def test_exact_match_is_case_and_punctuation_insensitive():
    assert exact_match("Business Administration", "business administration.") == 1.0
    assert exact_match("The Eiffel Tower", "eiffel tower") == 1.0  # article stripped
    assert exact_match("Boston", "Kochi") == 0.0


def test_token_f1_partial_credit_for_overlapping_but_not_identical_answers():
    f1 = token_f1("a degree in Business Administration", "Business Administration")
    assert 0.0 < f1 < 1.0
    assert token_f1("Boston", "Boston") == 1.0
    assert token_f1("Boston", "Kochi") == 0.0


def test_normalize_answer_strips_articles_and_punctuation():
    assert normalize_answer("The Business Administration, degree.") == "business administration degree"


def test_empty_prediction_scores_zero_against_a_real_reference():
    assert exact_match("", "Boston") == 0.0
    assert token_f1("", "Boston") == 0.0
