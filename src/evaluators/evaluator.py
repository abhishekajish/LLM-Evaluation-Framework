import re


def normalize_text(text):
    """
    Normalize text before comparison.

    This makes evaluation less sensitive to:
    - capitalization
    - punctuation
    - extra whitespace
    """

    text = text.lower().strip()

    text = re.sub(r"[^\w\s$%.-]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text


def exact_match(predicted, expected):
    """
    Compare the model response with the expected answer.

    Returns:
        1 -> correct
        0 -> incorrect
    """

    predicted = normalize_text(predicted)
    expected = normalize_text(expected)

    return int(predicted == expected)