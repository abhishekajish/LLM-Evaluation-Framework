def judge_response(
    question,
    context,
    expected_answer,
    model_response
):
    """
    Evaluate a model response using a deterministic
    rule-based judge.

    This acts as a placeholder for a real
    LLM-as-a-Judge implementation.
    """

    response = model_response.lower().strip()
    expected = expected_answer.lower().strip()

    # Basic correctness check
    if response == expected:
        correctness = 1.0

    elif expected in response:
        correctness = 0.8

    else:
        correctness = 0.0

    # Basic context faithfulness check
    if context:

        context_lower = context.lower()

        if expected in context_lower and expected in response:
            faithfulness = 1.0

        elif expected in response:
            faithfulness = 0.8

        else:
            faithfulness = 0.0

    else:
        faithfulness = correctness

    # Relevance is approximated using correctness
    relevance = correctness

    # Overall judge score
    overall = (
        0.4 * correctness
        + 0.3 * relevance
        + 0.3 * faithfulness
    )

    return {
        "correctness": round(correctness, 4),
        "relevance": round(relevance, 4),
        "faithfulness": round(faithfulness, 4),
        "judge_score": round(overall, 4)
    }