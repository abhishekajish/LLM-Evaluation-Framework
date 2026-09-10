import pandas as pd


def generate_leaderboard(results):
    """
    Generate overall model leaderboard.

    Composite score:
        30% Exact Match
        40% Semantic Similarity
        30% Judge Score
    """

    leaderboard = (
        results
        .groupby("model")
        .agg(
            exact_match=("exact_match", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            judge_score=("judge_score", "mean")
        )
    )

    leaderboard["composite_score"] = (
        0.3 * leaderboard["exact_match"]
        + 0.4 * leaderboard["semantic_similarity"]
        + 0.3 * leaderboard["judge_score"]
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
            judge_score=("judge_score", "mean")
        )
        .reset_index()
    )

    category_scores["composite_score"] = (
        0.3 * category_scores["exact_match"]
        + 0.4 * category_scores["semantic_similarity"]
        + 0.3 * category_scores["judge_score"]
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