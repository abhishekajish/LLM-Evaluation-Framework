import re
import math


def normalize_text(text):
    """
    Normalize text for comparison.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s$%.-]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

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
        r"\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|k))?",
        text
    )

    for value in monetary_values:
        value = re.sub(r"\s+", " ", value.strip())
        facts.add(value)

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
    Estimate whether factual information in a response
    is supported by the provided context.

    Returns:
        1.0 -> fully supported
        0.0 -> unsupported factual information
        NaN -> no context available
    """

    if not context:
        return math.nan

    context_normalized = normalize_text(context)
    response_normalized = normalize_text(response)

    if not response_normalized:
        return 0.0

    # ---------------------------------
    # Extract factual values
    # ---------------------------------

    context_facts = extract_facts(
        context_normalized
    )

    response_facts = extract_facts(
        response_normalized
    )

    unsupported_facts = (
        response_facts - context_facts
    )

    # Any unsupported factual value
    # indicates a potential hallucination.
    if unsupported_facts:
        return 0.0

    # If factual values exist and all
    # are supported, score as fully grounded.
    if response_facts:
        return 1.0

    # ---------------------------------
    # Lexical grounding
    # ---------------------------------

    response_words = set(
        response_normalized.split()
    )

    context_words = set(
        context_normalized.split()
    )

    if not response_words:
        return 0.0

    supported_words = (
        response_words.intersection(
            context_words
        )
    )

    score = (
        len(supported_words)
        / len(response_words)
    )

    return round(score, 4)