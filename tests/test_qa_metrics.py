"""EM/F1 scoring for the downstream QA eval -- SQuAD-style normalization
must actually behave as claimed (case/punctuation/article-insensitive)."""
from memorylife.evaluation.qa_metrics import bleu1, exact_match, is_refusal, normalize_answer, token_f1


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


def test_is_refusal_detects_real_gpt4o_refusal_phrasings():
    # calibrated against actual Week-6 predictions, not written blind
    assert is_refusal("I don't have that information yet.")
    assert is_refusal("You don't have that information yet.")
    assert is_refusal("$15 on car wash. I don't have information on a parking ticket.")
    assert is_refusal("James has pets. I don't have that information for John.")
    assert is_refusal("Sorry, I don't know.")
    assert is_refusal("Unable to determine from the retrieved memories.")


def test_is_refusal_does_not_false_positive_on_real_answers():
    assert not is_refusal("2023-05-07")
    assert not is_refusal("Auntie")
    assert not is_refusal("Gathering information and a beginners' guide.")
    assert not is_refusal("Kickboxing, taekwondo.")


def test_bleu1_is_1_for_identical_strings_and_0_for_disjoint_ones():
    assert bleu1("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert bleu1("completely different words here", "the cat sat on the mat") == 0.0


def test_bleu1_applies_a_brevity_penalty_for_a_short_but_precise_prediction():
    score = bleu1("cat mat", "the cat sat on the mat")
    assert 0.0 < score < 1.0  # perfect unigram precision, but penalized for being short


def test_bleu1_empty_prediction_scores_zero():
    assert bleu1("", "Boston") == 0.0
