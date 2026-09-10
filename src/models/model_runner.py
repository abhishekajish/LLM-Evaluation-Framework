import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


MODELS = {
    "qwen": "qwen3:4b"
}


def run_model(model_name, question, context=""):
    """
    Run a real local LLM through Ollama.
    """

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model = MODELS[model_name]

    # Build prompt
    if context:
        prompt = f"""
Context:
{context}

Question:
{question}

Answer the question using the provided context.
Give only the answer and do not add unnecessary explanation.
"""
    else:
        prompt = f"""
Question:
{question}

Give only the answer and do not add unnecessary explanation.
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()