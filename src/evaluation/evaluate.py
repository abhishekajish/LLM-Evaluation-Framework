import pandas as pd

from src.models.model_runner import run_model
from src.evaluators.evaluator import exact_match


def evaluate_model(model_name, dataset):
    """
    Evaluate one model against the complete dataset.
    """

    results = []

    for sample in dataset:

        # Run model
        response = run_model(
            model_name=model_name,
            question=sample["question"],
            context=sample["context"]
        )

        # Evaluate response
        score = exact_match(
            predicted=response,
            expected=sample["expected_answer"]
        )

        # Store result
        results.append({
            "id": sample["id"],
            "model": model_name,
            "category": sample["category"],
            "question": sample["question"],
            "expected_answer": sample["expected_answer"],
            "model_response": response,
            "score": score
        })

    return pd.DataFrame(results)