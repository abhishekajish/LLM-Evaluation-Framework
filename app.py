import os

import pandas as pd
import streamlit as st

from src.analysis.leaderboard import (
    generate_leaderboard,
    generate_category_scores,
    generate_robustness_scores,
    generate_failure_analysis
)


RESULTS_PATH = "data/processed/evaluation_results.csv"


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(
    page_title="LLM Evaluation Framework",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------
# Custom styling
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

        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #ddd;
            background-color: #fafafa;
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
# Load results
# ---------------------------------

if not os.path.exists(RESULTS_PATH):

    st.error(
        "No evaluation results found. "
        "Run the evaluation pipeline first."
    )

    st.stop()


results = pd.read_csv(RESULTS_PATH)


# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title("Evaluation Controls")

models = sorted(
    results["model"].unique()
)

selected_models = st.sidebar.multiselect(
    "Select Models",
    models,
    default=models
)

categories = sorted(
    results["category"].unique()
)

selected_categories = st.sidebar.multiselect(
    "Select Categories",
    categories,
    default=categories
)


filtered_results = results[
    results["model"].isin(selected_models)
    & results["category"].isin(selected_categories)
]


# ---------------------------------
# KPI cards
# ---------------------------------

st.subheader("Evaluation Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Models Evaluated",
        filtered_results["model"].nunique()
    )

with col2:
    st.metric(
        "Samples",
        len(filtered_results)
    )

with col3:
    st.metric(
        "Avg Semantic Similarity",
        f"{filtered_results['semantic_similarity'].mean():.1%}"
    )

with col4:
    if "latency_seconds" in filtered_results.columns:
        avg_latency = (
            filtered_results["latency_seconds"].mean()
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

st.subheader("🏆 Model Leaderboard")

if filtered_results.empty:

    st.warning("No data available for the selected filters.")

else:

    leaderboard = generate_leaderboard(
        filtered_results
    )

    display_leaderboard = leaderboard.copy()

    for column in [
        "exact_match",
        "semantic_similarity",
        "judge_score",
        "hallucination_score",
        "composite_score"
    ]:

        if column in display_leaderboard.columns:

            display_leaderboard[column] = (
                display_leaderboard[column] * 100
            ).round(2)

    if "composite_score" in display_leaderboard.columns:

        display_leaderboard = (
            display_leaderboard
            .rename(
                columns={
                    "exact_match": "Exact Match %",
                    "semantic_similarity": "Semantic Similarity %",
                    "judge_score": "Judge Score %",
                    "hallucination_score": "Hallucination Score %",
                    "composite_score": "Composite Score %"
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

st.subheader("📊 Category Performance")

category_scores = generate_category_scores(
    filtered_results
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

    st.bar_chart(chart_data)


# ---------------------------------
# Adversarial robustness
# ---------------------------------

st.divider()

st.subheader("🛡️ Adversarial Robustness")

robustness_scores = generate_robustness_scores(
    filtered_results
)

if robustness_scores.empty:

    st.info(
        "No adversarial samples available."
    )

else:

    robustness_display = robustness_scores[
        ["robustness_score"]
    ].copy()

    robustness_display[
        "robustness_score"
    ] = (
        robustness_display[
            "robustness_score"
        ] * 100
    ).round(2)

    robustness_display = (
        robustness_display
        .rename(
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

st.subheader("🔎 Failure Analysis")

failures = generate_failure_analysis(
    filtered_results
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
# Individual responses
# ---------------------------------

st.divider()

st.subheader("💬 Response Inspection")

if not filtered_results.empty:

    selected_model = st.selectbox(
        "Model",
        sorted(
            filtered_results["model"].unique()
        )
    )

    model_results = filtered_results[
        filtered_results["model"]
        == selected_model
    ]

    selected_id = st.selectbox(
        "Evaluation Sample",
        model_results["id"].tolist()
    )

    sample = model_results[
        model_results["id"] == selected_id
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Question**")

        st.info(
            sample["question"]
        )

        if "context" in sample.index and pd.notna(sample["context"]) and sample["context"]:

            st.markdown("**Context**")

            st.write(
                sample["context"]
            )

    with col2:

        st.markdown("**Expected Answer**")

        st.success(
            sample["expected_answer"]
        )

        st.markdown("**Model Response**")

        st.write(
            sample["model_response"]
        )

    st.markdown("**Evaluation Metrics**")

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

    metrics_df = pd.DataFrame({
        "Metric": available_metrics,
        "Value": [
            sample[column]
            for column in available_metrics
        ]
    })

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------
# Raw results
# ---------------------------------

with st.expander("View Raw Evaluation Results"):

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