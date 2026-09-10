import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from src.analysis.leaderboard import (
    generate_leaderboard,
    generate_category_scores,
    generate_robustness_scores,
    generate_failure_analysis
)


RESULTS_PATH = "data/processed/evaluation_results.csv"
HISTORY_DIR = "results"


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(
    page_title="LLM Evaluation Framework",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------
# Styling
# ---------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            color: #777;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Header
# ---------------------------------

st.markdown(
    '<div class="main-title">🤖 LLM Evaluation Framework</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Local, modular evaluation and comparison of Large Language Models'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------
# Load existing results
# ---------------------------------

if os.path.exists(RESULTS_PATH):

    results = pd.read_csv(
        RESULTS_PATH
    )

else:

    results = pd.DataFrame()


# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title(
    "Evaluation Controls"
)


# ---------------------------------
# Model selection
# ---------------------------------

available_models = [
    "qwen",
    "llama"
]

selected_models = st.sidebar.multiselect(
    "Select Models",
    available_models,
    default=available_models
)


# ---------------------------------
# Dataset upload
# ---------------------------------

st.sidebar.subheader(
    "Dataset"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload evaluation dataset",
    type=["json"]
)


evaluation_dataset = None


if uploaded_file is not None:

    try:

        evaluation_dataset = json.load(
            uploaded_file
        )

        if not isinstance(
            evaluation_dataset,
            list
        ):

            st.sidebar.error(
                "Dataset must contain a JSON list."
            )

            evaluation_dataset = None

        else:

            st.sidebar.success(
                f"{len(evaluation_dataset)} samples loaded."
            )

    except Exception as error:

        st.sidebar.error(
            f"Invalid JSON: {error}"
        )


# ---------------------------------
# Run evaluation
# ---------------------------------

run_evaluation = st.sidebar.button(
    "🚀 Run Evaluation",
    use_container_width=True
)


if run_evaluation:

    if not selected_models:

        st.error(
            "Select at least one model."
        )

        st.stop()


    if evaluation_dataset is None:

        st.error(
            "Upload a JSON evaluation dataset first."
        )

        st.stop()


    from src.evaluation.evaluate import (
        evaluate_model
    )


    evaluation_results = []


    progress = st.progress(
        0
    )

    status = st.empty()


    for index, model in enumerate(
        selected_models
    ):

        status.write(
            f"Evaluating {model}..."
        )


        model_results = evaluate_model(
            model_name=model,
            dataset=evaluation_dataset
        )


        evaluation_results.append(
            model_results
        )


        progress.progress(
            (index + 1)
            / len(selected_models)
        )


    final_evaluation = pd.concat(
        evaluation_results,
        ignore_index=True
    )


    # ---------------------------------
    # Save latest results
    # ---------------------------------

    os.makedirs(
        os.path.dirname(RESULTS_PATH),
        exist_ok=True
    )


    final_evaluation.to_csv(
        RESULTS_PATH,
        index=False
    )


    # ---------------------------------
    # Save evaluation history
    # ---------------------------------

    os.makedirs(
        HISTORY_DIR,
        exist_ok=True
    )


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    history_path = os.path.join(
        HISTORY_DIR,
        f"evaluation_{timestamp}.csv"
    )


    final_evaluation.to_csv(
        history_path,
        index=False
    )


    status.success(
        "Evaluation completed successfully."
    )


    st.success(
        f"Results saved successfully. "
        f"History: {history_path}"
    )


    st.rerun()


# ---------------------------------
# No results available
# ---------------------------------

if results.empty:

    st.warning(
        "No evaluation results are available yet."
    )

    st.info(
        "Upload a JSON dataset and click "
        "'Run Evaluation' to begin."
    )

    st.stop()


# ---------------------------------
# Category filters
# ---------------------------------

models_in_results = sorted(
    results["model"].unique()
)


selected_models = [
    model
    for model in selected_models
    if model in models_in_results
]


categories = sorted(
    results["category"].unique()
)


selected_categories = st.sidebar.multiselect(
    "Select Categories",
    categories,
    default=categories
)


# ---------------------------------
# Filter results
# ---------------------------------

filtered_results = results[
    results["model"].isin(
        selected_models
    )
    &
    results["category"].isin(
        selected_categories
    )
]


# ---------------------------------
# KPI cards
# ---------------------------------

st.subheader(
    "Evaluation Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Models Evaluated",
        filtered_results[
            "model"
        ].nunique()
    )


with col2:

    st.metric(
        "Samples",
        len(filtered_results)
    )


with col3:

    if not filtered_results.empty:

        semantic_average = (
            filtered_results[
                "semantic_similarity"
            ].mean()
        )

        st.metric(
            "Avg Semantic Similarity",
            f"{semantic_average:.1%}"
        )

    else:

        st.metric(
            "Avg Semantic Similarity",
            "N/A"
        )


with col4:

    if (
        "latency_seconds"
        in filtered_results.columns
        and not filtered_results.empty
    ):

        avg_latency = (
            filtered_results[
                "latency_seconds"
            ].mean()
        )

        st.metric(
            "Avg Latency",
            f"{avg_latency:.2f}s"
        )

    else:

        st.metric(
            "Avg Latency",
            "Not available"
        )


# ---------------------------------
# Leaderboard
# ---------------------------------

st.divider()

st.subheader(
    "🏆 Model Leaderboard"
)


if filtered_results.empty:

    st.warning(
        "No data available for the selected filters."
    )

else:

    leaderboard = generate_leaderboard(
        filtered_results
    )


    display_leaderboard = (
        leaderboard.copy()
    )


    percentage_columns = [
        "exact_match",
        "semantic_similarity",
        "judge_score",
        "hallucination_score",
        "composite_score"
    ]


    for column in percentage_columns:

        if column in display_leaderboard.columns:

            display_leaderboard[
                column
            ] = (
                display_leaderboard[
                    column
                ] * 100
            ).round(2)


    display_leaderboard = (
        display_leaderboard.rename(
            columns={
                "exact_match":
                    "Exact Match %",
                "semantic_similarity":
                    "Semantic Similarity %",
                "judge_score":
                    "Judge Score %",
                "hallucination_score":
                    "Hallucination Score %",
                "composite_score":
                    "Composite Score %"
            }
        )
    )


    st.dataframe(
        display_leaderboard,
        use_container_width=True
    )


# ---------------------------------
# Category performance
# ---------------------------------

st.divider()

st.subheader(
    "📊 Category Performance"
)


category_scores = (
    generate_category_scores(
        filtered_results
    )
)


if not category_scores.empty:

    chart_data = (
        category_scores
        .pivot(
            index="category",
            columns="model",
            values="composite_score"
        )
    )


    st.bar_chart(
        chart_data
    )


# ---------------------------------
# Adversarial robustness
# ---------------------------------

st.divider()

st.subheader(
    "🛡️ Adversarial Robustness"
)


robustness_scores = (
    generate_robustness_scores(
        filtered_results
    )
)


if robustness_scores.empty:

    st.info(
        "No adversarial samples available."
    )

else:

    robustness_display = (
        robustness_scores[
            ["robustness_score"]
        ].copy()
    )


    robustness_display[
        "robustness_score"
    ] = (
        robustness_display[
            "robustness_score"
        ] * 100
    ).round(2)


    robustness_display = (
        robustness_display.rename(
            columns={
                "robustness_score":
                    "Robustness Score %"
            }
        )
    )


    st.dataframe(
        robustness_display,
        use_container_width=True
    )


# ---------------------------------
# Failure analysis
# ---------------------------------

st.divider()

st.subheader(
    "🔎 Failure Analysis"
)


failures = (
    generate_failure_analysis(
        filtered_results
    )
)


if failures.empty:

    st.success(
        "No evaluation failures detected."
    )

else:

    st.dataframe(
        failures,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------
# Response inspection
# ---------------------------------

st.divider()

st.subheader(
    "💬 Response Inspection"
)


if not filtered_results.empty:

    selected_model_for_inspection = (
        st.selectbox(
            "Model",
            sorted(
                filtered_results[
                    "model"
                ].unique()
            )
        )
    )


    model_results = (
        filtered_results[
            filtered_results[
                "model"
            ]
            == selected_model_for_inspection
        ]
    )


    selected_id = st.selectbox(
        "Evaluation Sample",
        model_results["id"].tolist()
    )


    sample = (
        model_results[
            model_results["id"]
            == selected_id
        ]
        .iloc[0]
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "**Question**"
        )

        st.info(
            sample["question"]
        )


        if (
            "context"
            in sample.index
            and pd.notna(
                sample["context"]
            )
            and sample["context"]
        ):

            st.markdown(
                "**Context**"
            )

            st.write(
                sample["context"]
            )


    with col2:

        st.markdown(
            "**Expected Answer**"
        )

        st.success(
            sample["expected_answer"]
        )


        st.markdown(
            "**Model Response**"
        )

        st.write(
            sample["model_response"]
        )


    st.markdown(
        "**Evaluation Metrics**"
    )


    metric_columns = [
        "exact_match",
        "semantic_similarity",
        "judge_correctness",
        "judge_relevance",
        "judge_faithfulness",
        "judge_score",
        "hallucination_score",
        "latency_seconds"
    ]


    available_metrics = [
        column
        for column in metric_columns
        if column in sample.index
    ]


    metrics_df = pd.DataFrame(
        {
            "Metric":
                available_metrics,
            "Value":
                [
                    sample[column]
                    for column
                    in available_metrics
                ]
        }
    )


    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------
# Evaluation history
# ---------------------------------

st.divider()

st.subheader(
    "🕘 Evaluation History"
)


if os.path.exists(HISTORY_DIR):

    history_files = sorted(
        [
            file
            for file in os.listdir(
                HISTORY_DIR
            )
            if file.endswith(".csv")
        ],
        reverse=True
    )

else:

    history_files = []


if not history_files:

    st.info(
        "No previous evaluation runs found."
    )

else:

    history_data = []


    for filename in history_files:

        filepath = os.path.join(
            HISTORY_DIR,
            filename
        )


        history_df = pd.read_csv(
            filepath
        )


        history_data.append(
            {
                "Run": filename,
                "Models": ", ".join(
                    sorted(
                        history_df[
                            "model"
                        ].unique()
                    )
                ),
                "Samples": len(
                    history_df
                ),
                "Avg Semantic Similarity":
                    round(
                        history_df[
                            "semantic_similarity"
                        ].mean() * 100,
                        2
                    ),
                "Avg Latency":
                    (
                        round(
                            history_df[
                                "latency_seconds"
                            ].mean(),
                            2
                        )
                        if "latency_seconds"
                        in history_df.columns
                        else None
                    )
            }
        )


    history_table = pd.DataFrame(
        history_data
    )


    st.dataframe(
        history_table,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------
# Raw results
# ---------------------------------

with st.expander(
    "View Raw Evaluation Results"
):

    st.dataframe(
        filtered_results,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "LLM Evaluation Framework • "
    "Local inference with Ollama • "
    "₹0 cloud/API cost"
)