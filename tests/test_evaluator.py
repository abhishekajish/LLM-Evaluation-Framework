from src.evaluators.evaluator import exact_match


def test_exact_match_correct():
    assert exact_match(
        "Paris",
        "Paris"
    ) == 1


def test_exact_match_incorrect():
    assert exact_match(
        "London",
        "Paris"
    ) == 0


def test_exact_match_case_insensitive():
    assert exact_match(
        "PARIS",
        "paris"
    ) == 1