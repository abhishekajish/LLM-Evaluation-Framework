import pandas as pd


def calculate_composite_score(data):
    """
    Calculate a weighted composite score.

    Weights:
        25% Exact Match
        30% Semantic Similarity
        25% Judge Score
        20% Hallucination Score

    If a metric is not applicable (NaN), its weight
    is excluded and the remaining weights are normalized.
    """

    metrics = {
        "exact_match": 0.25,
        "semantic_similarity": 0.30,
        "judge_score": 0.25,
        "hallucination_score": 0.20
    }

    weighted_score = 0.0
    total_weight = 0.0

    for metric, weight in metrics.items():

        if pd.notna(data[metric]):
            weighted_score += data[metric] * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_score / total_weight


def generate_leaderboard(results):
    """
    Generate overall model leaderboard.
    """

    leaderboard = (
        results
        .groupby("model")
        .agg(
            exact_match=("exact_match", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            judge_score=("judge_score", "mean"),
            hallucination_score=("hallucination_score", "mean")
        )
    )

    # Calculate composite using normalized available metrics.
    leaderboard["composite_score"] = leaderboard.apply(
        calculate_composite_score,
        axis=1
    )

    leaderboard = leaderboard.sort_values(
        "composite_score",
        ascending=False
    )

    return leaderboard


def generate_category_scores(results):
    """
    Generate model performance grouped by category.
    """

    category_scores = (
        results
        .groupby(["model", "category"])
        .agg(
            exact_match=("exact_match", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            judge_score=("judge_score", "mean"),
            hallucination_score=("hallucination_score", "mean")
        )
        .reset_index()
    )

    # Calculate composite using only applicable metrics.
    category_scores["composite_score"] = category_scores.apply(
        calculate_composite_score,
        axis=1
    )

    return category_scores


if __name__ == "__main__":

    results = pd.read_csv(
        "data/processed/evaluation_results.csv"
    )

    print("\nMODEL LEADERBOARD:\n")

    leaderboard = generate_leaderboard(results)

    print(leaderboard)

    print("\n\nCATEGORY PERFORMANCE:\n")

    category_scores = generate_category_scores(results)

    print(category_scores.to_string(index=False))