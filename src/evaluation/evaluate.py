import pandas as pd

from src.models.model_runner import run_model
from src.evaluators.evaluator import (
    exact_match,
    semantic_similarity
)
from src.evaluators.judge import judge_response
from src.evaluators.hallucination import hallucination_score


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

        # Exact match evaluation
        exact_score = exact_match(
            predicted=response,
            expected=sample["expected_answer"]
        )

        # Semantic similarity evaluation
        semantic_score = semantic_similarity(
            predicted=response,
            expected=sample["expected_answer"]
        )

        # Judge evaluation
        judge_scores = judge_response(
            question=sample["question"],
            context=sample["context"],
            expected_answer=sample["expected_answer"],
            model_response=response
        )

        # Hallucination evaluation
        hallucination = hallucination_score(
            context=sample["context"],
            response=response
        )

        # Store result
        results.append({
            "id": sample["id"],
            "model": model_name,
            "category": sample["category"],
            "question": sample["question"],
            "expected_answer": sample["expected_answer"],
            "model_response": response,
            "exact_match": exact_score,
            "semantic_similarity": semantic_score,
            "judge_correctness": judge_scores["correctness"],
            "judge_relevance": judge_scores["relevance"],
            "judge_faithfulness": judge_scores["faithfulness"],
            "judge_score": judge_scores["judge_score"],
            "hallucination_score": hallucination
        })

    return pd.DataFrame(results)