import altair as alt
import numpy as np
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


def plot_best_by_recall_score(df: pd.DataFrame):
    df_recall = df.loc[df.groupby("experiment_id")["recall_1"].idxmax()]
    df_recall["experiment_id"] = df_recall["experiment_id"].astype(int)
    df_recall.sort_values(by="experiment_id", inplace=True)
    df_recall["experiment_idx"] = np.arange(1, len(df_recall) + 1)

    st.altair_chart(
        alt.Chart(
            data=df_recall,
            title=alt.TitleParams(
                text="Best Models by Recall Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("recall_1:Q", title="Recall Score"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
        theme=None,
    )

    st.dataframe(df_recall)


def plot_worst_by_recall_score(df: pd.DataFrame):
    df_recall_min = df.loc[df.groupby("experiment_id")["recall_1"].idxmin()]
    df_recall_min["experiment_id"] = df_recall_min["experiment_id"].astype(int)
    df_recall_min.sort_values(by="experiment_id", inplace=True)
    df_recall_min["experiment_idx"] = np.arange(1, len(df_recall_min) + 1)

    st.altair_chart(
        alt.Chart(
            data=df_recall_min,
            title=alt.TitleParams(
                text="Worst Models by Recall Score Across Experiments",
                anchor="start",
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X("experiment_idx:N", title="Experiments").sort("x"),
            y=alt.Y("recall_1:Q", title="Recall Score"),
            color=alt.Color("parent_run_name:N", title="Models"),
        ),
        use_container_width=True,
        theme=None,
    )

    st.dataframe(df_recall_min)


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
