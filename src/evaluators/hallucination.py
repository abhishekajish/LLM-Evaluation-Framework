import re
import math


def normalize_text(text):
    """
    Normalize text for comparison.
    """

    text = text.lower().strip()

    text = re.sub(r"[^\w\s$%.-]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text


def extract_facts(text):
    """
    Extract important factual values such as:
    - monetary values
    - percentages
    - standalone numbers
    """

    text = text.lower()

    facts = set()

    # ---------------------------------
    # Monetary values
    # ---------------------------------

    monetary_values = re.findall(
        r"\$\s*\d+(?:\.\d+)?",
        text
    )

    for value in monetary_values:
        facts.add(
            value.replace(" ", "")
        )

    # ---------------------------------
    # Percentages
    # ---------------------------------

    percentages = re.findall(
        r"\d+(?:\.\d+)?\s*%",
        text
    )

    for value in percentages:
        facts.add(
            value.replace(" ", "")
        )

    # ---------------------------------
    # Standalone numbers
    # ---------------------------------

    numbers = re.findall(
        r"(?<![\w$])\d+(?:\.\d+)?(?![\w%])",
        text
    )

    facts.update(numbers)

    return facts


def hallucination_score(context, response):
    """
    Estimate how well a model response is supported
    by the supplied context.

    Returns:
        1.0 -> response is fully supported
        0.0 -> response contains unsupported factual values
    """

    if not context:
        return math.nan

    context = normalize_text(context)
    response = normalize_text(response)

    if not response:
        return 0.0

    # ---------------------------------
    # Check factual values
    # ---------------------------------

    context_facts = extract_facts(context)
    response_facts = extract_facts(response)

    unsupported_facts = response_facts - context_facts

    # Any new factual value not found in the
    # context is treated as a hallucination.
    if unsupported_facts:
        return 0.0

    # ---------------------------------
    # Factual values are supported
    # ---------------------------------

    if response_facts:
        return 1.0

    # ---------------------------------
    # No factual values detected
    # ---------------------------------

    response_words = set(response.split())
    context_words = set(context.split())

    supported_words = response_words.intersection(
        context_words
    )

    score = len(supported_words) / len(response_words)

    return round(score, 4)