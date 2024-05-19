import altair as alt
import pandas as pd
import streamlit as st

model_names = [
    "KNN",
    "Logistic Regression",
    "Gaussian NB",
    "Linear SVC",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def model_selector(model_names):
    cols = st.columns([1, 1, 1])

    return cols[0].selectbox(label="Models", options=model_names)


def create_model_selector():
    return model_selector(model_names)


def plot_recall_score_by_experiments(df_cv_metrics: pd.DataFrame):
    return st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title=alt.TitleParams(
                text="Sampling and Preprocessing Techniques by Recall Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(
                shorthand="experiment_idx:N",
                title="Experiments",
                axis=alt.Axis(labelAngle=0),
                sort=alt.Sort("x"),
            ),
            y=alt.Y("recall_1:Q", title="Recall"),
        ),
        use_container_width=True,
        theme=None,
    )
