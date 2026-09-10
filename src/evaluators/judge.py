import requests
import json

from src.evaluators.evaluator import normalize_text


OLLAMA_URL = "http://localhost:11434/api/generate"

JUDGE_MODEL = "gemma3:1b"


def judge_response(
    question,
    context,
    expected_answer,
    model_response
):
    """
    Evaluate a model response using a local LLM judge
    with deterministic guardrails for exact answers.
    """

    normalized_expected = normalize_text(
        expected_answer
    )

    normalized_response = normalize_text(
        model_response
    )

    if normalized_expected == normalized_response:
        return {
            "correctness": 1.0,
            "relevance": 1.0,
            "faithfulness": 1.0,
            "judge_score": 1.0
        }

    prompt = f"""
You are a strict but fair evaluator of an AI model response.

Compare the MODEL RESPONSE against the EXPECTED ANSWER.

QUESTION:
{question}

CONTEXT:
{context}

EXPECTED ANSWER:
{expected_answer}

MODEL RESPONSE:
{model_response}

SCORING RULES:

CORRECTNESS:
- 1.0 = substantively correct answer.
- 0.5 = partially correct answer.
- 0.0 = incorrect answer.
- Do not require explanations unless explicitly requested.
- Ignore capitalization and minor punctuation differences.

RELEVANCE:
- 1.0 = directly answers the question.
- 0.5 = partially addresses the question.
- 0.0 = unrelated or answers a different question.

FAITHFULNESS:
- 1.0 = supported by the supplied context.
- 0.5 = partially supported.
- 0.0 = contradicts or invents information.
- If no context is provided, judge consistency with the expected answer.

IMPORTANT:
- Short factual answers can receive 1.0.
- Do not penalize concise answers.
- Do not penalize punctuation.
- Do not penalize capitalization.

Return ONLY valid JSON:

{{
    "correctness": 1.0,
    "relevance": 1.0,
    "faithfulness": 1.0
}}
"""

    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    judge_result = json.loads(
        result["response"]
    )

    correctness = float(
        judge_result.get("correctness", 0.0)
    )

    relevance = float(
        judge_result.get("relevance", 0.0)
    )

    faithfulness = float(
        judge_result.get("faithfulness", 0.0)
    )

    correctness = max(0.0, min(1.0, correctness))
    relevance = max(0.0, min(1.0, relevance))
    faithfulness = max(0.0, min(1.0, faithfulness))

    judge_score = (
        0.4 * correctness
        + 0.3 * relevance
        + 0.3 * faithfulness
    )

    return {
        "correctness": round(correctness, 4),
        "relevance": round(relevance, 4),
        "faithfulness": round(faithfulness, 4),
        "judge_score": round(judge_score, 4)
    }
