from src.data.loader import load_dataset
from src.evaluation.evaluate import evaluate_model
import pandas as pd


DATASET_PATH = "data/raw/evaluation_dataset.json"

MODELS = [
    "qwen",
    "llama",
    "llama70b"
]


def main():

    print("\n" + "=" * 50)
    print("       LLM EVALUATION FRAMEWORK")
    print("=" * 50)

    # -----------------------------
    # Load dataset
    # -----------------------------

    print("\nLoading evaluation dataset...")

    dataset = load_dataset(DATASET_PATH)

    print(f"Loaded {len(dataset)} evaluation samples.")

    # -----------------------------
    # Evaluate models
    # -----------------------------

    all_results = []

    for model in MODELS:

        print(f"\nEvaluating model: {model}")

        results = evaluate_model(
            model_name=model,
            dataset=dataset
        )

        all_results.append(results)

        accuracy = results["score"].mean()

        print(
            f"{model} accuracy: "
            f"{accuracy:.2%}"
        )

    # -----------------------------
    # Combine results
    # -----------------------------

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    # -----------------------------
    # Save results
    # -----------------------------

    output_path = (
        "data/processed/evaluation_results.csv"
    )

    final_results.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 50)
    print("Evaluation completed successfully.")
    print(f"Results saved to: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()