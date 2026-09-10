import pandas as pd


def calculate_composite_score(data):
    """
    Calculate a weighted composite score.

    Weights:
        25% Exact Match
        30% Semantic Similarity
        25% Judge Score
        20% Hallucination Score

    Metrics with NaN values are excluded and
    remaining weights are normalized.
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

            weighted_score += (
                data[metric] * weight
            )

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

    leaderboard["composite_score"] = (
        leaderboard.apply(
            calculate_composite_score,
            axis=1
        )
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
        .groupby(
            ["model", "category"]
        )
        .agg(
            exact_match=("exact_match", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            judge_score=("judge_score", "mean"),
            hallucination_score=("hallucination_score", "mean")
        )
        .reset_index()
    )

    category_scores["composite_score"] = (
        category_scores.apply(
            calculate_composite_score,
            axis=1
        )
    )

    return category_scores


def generate_robustness_scores(results):
    """
    Generate performance scores for adversarial samples.
    """

    adversarial = results[
        results["category"] == "adversarial"
    ].copy()

    if adversarial.empty:
        return pd.DataFrame()

    robustness = (
        adversarial
        .groupby("model")
        .agg(
            exact_match=("exact_match", "mean"),
            semantic_similarity=("semantic_similarity", "mean"),
            judge_score=("judge_score", "mean"),
            hallucination_score=("hallucination_score", "mean")
        )
    )

    robustness["robustness_score"] = (
        0.30 * robustness["exact_match"]
        + 0.30 * robustness["semantic_similarity"]
        + 0.25 * robustness["judge_score"]
        + 0.15 * robustness["hallucination_score"].fillna(0)
    )

    return robustness.sort_values(
        "robustness_score",
        ascending=False
    )


def generate_failure_analysis(results):
    """
    Identify individual evaluation failures.

    A sample is flagged when one or more
    evaluation metrics indicate poor performance.
    """

    failures = results.copy()

    failures["failure_type"] = ""

    for index, row in failures.iterrows():

        failure_types = []

        if row["exact_match"] == 0:
            failure_types.append(
                "exact_match"
            )

        if row["semantic_similarity"] < 0.5:
            failure_types.append(
                "low_semantic_similarity"
            )

        if row["judge_score"] < 0.5:
            failure_types.append(
                "low_judge_score"
            )

        if (
            pd.notna(row["hallucination_score"])
            and row["hallucination_score"] == 0
        ):
            failure_types.append(
                "hallucination"
            )

        if (
            row["category"] == "adversarial"
            and row["exact_match"] == 0
        ):
            failure_types.append(
                "adversarial_failure"
            )

        failures.at[
            index,
            "failure_type"
        ] = ", ".join(failure_types)

    failures = failures[
        failures["failure_type"] != ""
    ]

    return failures[
        [
            "id",
            "model",
            "category",
            "question",
            "expected_answer",
            "model_response",
            "failure_type"
        ]
    ]


if __name__ == "__main__":

    results = pd.read_csv(
        "data/processed/evaluation_results.csv"
    )

    print("\nMODEL LEADERBOARD:\n")

    leaderboard = generate_leaderboard(
        results
    )

    print(leaderboard)

    print(
        "\n\nCATEGORY PERFORMANCE:\n"
    )

    category_scores = (
        generate_category_scores(
            results
        )
    )

    print(
        category_scores.to_string(
            index=False
        )
    )

    print(
        "\n\nADVERSARIAL ROBUSTNESS:\n"
    )

    robustness_scores = (
        generate_robustness_scores(
            results
        )
    )

    if robustness_scores.empty:
        print(
            "No adversarial samples found."
        )
    else:
        print(
            robustness_scores.to_string()
        )

    print(
        "\n\nFAILURE ANALYSIS:\n"
    )

    failures = generate_failure_analysis(
        results
    )

    if failures.empty:
        print(
            "No failures detected."
        )
    else:
        print(
            failures.to_string(
                index=False
            )
        )