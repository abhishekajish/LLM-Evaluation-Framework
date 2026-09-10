import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"

JUDGE_MODEL = "gemma3:1b"


def judge_response(
    question,
    context,
    expected_answer,
    model_response
):
    """
    Evaluate a model response using a local LLM judge.

    The judge evaluates:
    - correctness
    - relevance
    - faithfulness

    Gemma 3 1B is used locally through Ollama.
    """

    prompt = f"""
You are an objective LLM evaluator.

Evaluate the model response against the question,
expected answer, and context.

Question:
{question}

Context:
{context}

Expected Answer:
{expected_answer}

Model Response:
{model_response}

Score the response on these dimensions:

1. correctness
2. relevance
3. faithfulness

Each score must be between 0 and 1.

Return ONLY valid JSON in exactly this format:

{{
    "correctness": 0.0,
    "relevance": 0.0,
    "faithfulness": 0.0
}}

Do not include explanations.
Do not include markdown.
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

    # Keep scores safely within 0-1.
    correctness = max(
        0.0,
        min(1.0, correctness)
    )

    relevance = max(
        0.0,
        min(1.0, relevance)
    )

    faithfulness = max(
        0.0,
        min(1.0, faithfulness)
    )

    # Weighted overall judge score.
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