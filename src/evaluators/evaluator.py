import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def semantic_similarity(predicted, expected):
    """
    Calculate semantic similarity between the model response
    and expected answer using TF-IDF cosine similarity.

    Returns:
        float between 0 and 1
    """

    predicted = normalize_text(predicted)
    expected = normalize_text(expected)

    # Handle identical answers directly
    if predicted == expected:
        return 1.0

    # Handle empty responses
    if not predicted or not expected:
        return 0.0

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b"
    )

    vectors = vectorizer.fit_transform(
        [predicted, expected]
    )

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0][0]

    return round(float(similarity), 4)