import altair as alt
import pandas as pd
import streamlit as st

from evaluate import evaluate_model
from predict import predict_image

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef4f8 100%);
    }
    .hero-card,
    .result-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 1.35rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }
    .hero-title {
        color: #0f2747;
        font-size: 2.15rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-text {
        color: #52637d;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .metric-label {
        color: #66768f;
        font-size: 0.92rem;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        color: #102542;
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Brain Tumor Detection with Grad-CAM</div>
        <p class="hero-text">
            Upload an MRI image to view the prediction result, class probabilities,
            original scan, and Grad-CAM explanation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def build_probability_chart(probabilities):
    chart_data = pd.DataFrame(
        {
            "Class": list(probabilities.keys()),
            "Probability": [score * 100 for score in probabilities.values()],
        }
    ).sort_values("Probability", ascending=False)

    color_scale = alt.Scale(
        domain=["glioma", "meningioma", "notumor", "pituitary"],
        range=["#ef4444", "#f59e0b", "#10b981", "#6366f1"],
    )

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X("Probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Class:N", sort="-x", title=None),
            color=alt.Color("Class:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("Class:N", title="Class"),
                alt.Tooltip("Probability:Q", title="Probability", format=".2f"),
            ],
        )
        .properties(height=260)
    )


def build_confusion_matrix_chart(confusion_df):
    heatmap_data = confusion_df.reset_index().melt(
        id_vars="index",
        var_name="Predicted",
        value_name="Count",
    ).rename(columns={"index": "Actual"})

    base = alt.Chart(heatmap_data).encode(
        x=alt.X("Predicted:N", title="Predicted Label"),
        y=alt.Y("Actual:N", title="Actual Label"),
    )

    heatmap = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
        tooltip=[
            alt.Tooltip("Actual:N", title="Actual"),
            alt.Tooltip("Predicted:N", title="Predicted"),
            alt.Tooltip("Count:Q", title="Count"),
        ],
    )

    text = base.mark_text(fontSize=14, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > heatmap_data["Count"].max() * 0.45,
            alt.value("white"),
            alt.value("#102542"),
        ),
    )

    return (heatmap + text).properties(height=320)


uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
)

st.subheader("Model Evaluation")
with st.spinner("Computing model metrics from the test dataset..."):
    evaluation = evaluate_model()

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_values = [
    ("Accuracy", evaluation["accuracy"] * 100),
    ("Precision", evaluation["precision"] * 100),
    ("Recall", evaluation["recall"] * 100),
    ("F1 Score", evaluation["f1"] * 100),
]

for column, (label, value) in zip(
    [metric_col1, metric_col2, metric_col3, metric_col4],
    metric_values,
):
    with column:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

matrix_col, report_col = st.columns([1.35, 1])
with matrix_col:
    st.subheader("Confusion Matrix")
    st.altair_chart(
        build_confusion_matrix_chart(evaluation["confusion_matrix"]),
        use_container_width=True,
    )

with report_col:
    st.subheader("Per-Class Metrics")
    st.dataframe(
        evaluation["class_report"],
        use_container_width=True,
        hide_index=True,
    )

if uploaded_file is not None:
    try:
        result = predict_image(uploaded_file.getvalue())
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Prediction failed: {error}")
    else:
        predicted_label = result["class_name"].replace("notumor", "No Tumor").title()
        confidence_percent = result["confidence"] * 100
        status_text = "No Tumor Detected" if result["class_name"] == "notumor" else "Tumor Detected"

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">Predicted Class</div>
                    <div class="metric-value">{predicted_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with summary_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence_percent:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with summary_col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">Status</div>
                    <div class="metric-value">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        chart_col, table_col = st.columns([1.6, 1])
        probability_df = pd.DataFrame(
            {
                "Class": list(result["probabilities"].keys()),
                "Probability": [f"{score * 100:.2f}%" for score in result["probabilities"].values()],
            }
        )

        with chart_col:
            st.subheader("Class Probability Chart")
            st.altair_chart(
                build_probability_chart(result["probabilities"]),
                use_container_width=True,
            )

        with table_col:
            st.subheader("Probability Table")
            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True,
            )

        if result["class_name"] == "notumor":
            st.success("No tumor detected.")
            st.image(
                result["original_image"],
                caption="Original Image",
                use_container_width=True,
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    result["original_image"],
                    caption="Original Image",
                    use_container_width=True,
                )
            with col2:
                st.image(
                    result["gradcam_image"],
                    caption="Grad-CAM Result",
                    use_container_width=True,
                )
else:
    st.info("Please upload an image to begin.")
