import pandas as pd


def generate_leaderboard(results):

    leaderboard = (
        results
        .groupby("model")["score"]
        .mean()
        .sort_values(ascending=False)
    )

    return leaderboard


if __name__ == "__main__":

    results = pd.read_csv(
        "data/processed/evaluation_results.csv"
    )

    leaderboard = generate_leaderboard(results)

    print("\nMODEL LEADERBOARD:\n")
    print(leaderboard)