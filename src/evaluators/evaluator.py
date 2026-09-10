import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text):
    """
    Normalize text before comparison.

    The normalization removes superficial formatting
    differences while preserving meaningful content.
    """

    text = text.lower().strip()

    # Remove surrounding quotation marks and punctuation.
    text = re.sub(
        r'^[\s\'"“”‘’.,!?;:()\[\]{}]+|[\s\'"“”‘’.,!?;:()\[\]{}]+$',
        '',
        text
    )

    # Remove punctuation that does not affect meaning.
    text = re.sub(
        r"[^\w\s$%.-]",
        "",
        text
    )

    # Remove trailing sentence punctuation.
    text = text.rstrip(".,!?;:")

    # Normalize whitespace.
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

    # Identical answers.
    if predicted == expected:
        return 1.0

    # Empty responses.
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