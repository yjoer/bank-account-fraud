import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from crunchy_mining.pages.fragments import _create_formatted_values
from crunchy_mining.pages.fragments import experiment_selector

experiments = [
    "clf/sampling_v1",
    "clf/sampling_v2",
    "clf/sampling_v3",
    "clf/sampling_v4",
    "clf/sampling_v5",
    "clf/sampling_v6",
    "clf/preprocessing_v1",
    "clf/preprocessing_v2",
    "clf/preprocessing_v3",
    "clf/preprocessing_v4",
    "clf/preprocessing_v5",
    "clf/preprocessing_v6",
    "clf/preprocessing_v7",
    "clf/resampling_v1",
    "clf/resampling_v2",
    "clf/resampling_v3",
    "clf/resampling_v4",
    "clf/resampling_v5",
    "clf/resampling_v6",
    "clf/resampling_v7",
    "clf/resampling_v8",
]

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

folds = {
    "validation": "Validation",
    "fold_1": "Fold 1",
    "fold_2": "Fold 2",
    "fold_3": "Fold 3",
    "fold_4": "Fold 4",
    "fold_5": "Fold 5",
}


def model_selector(model_names):
    cols = st.columns([1, 1, 1])

    return cols[0].selectbox(label="Models", options=model_names)


def create_model_selector():
    return model_selector(model_names)


def create_experiment_selector():
    return experiment_selector(experiments)


def create_experiment_model_selector():
    cols = st.columns([1, 1, 1])
    experiment = cols[0].selectbox(label="Experiments", options=experiments, key=1)

    model = cols[1].selectbox(label="Models", options=model_names, key=2)

    return experiment, model


def create_fold_selector():
    cols = st.columns([1, 1, 1])
    experiment = cols[0].selectbox(label="Experiments", options=experiments, key=3)

    model = cols[1].selectbox(label="Models", options=model_names, key=4)

    fold = cols[2].selectbox(
        label="Folds",
        options=folds.keys(),
        format_func=lambda x: folds[x],
    )

    return experiment, model, fold


def plot_best_by_recall_score(df: pd.DataFrame):
    df_recall = df.loc[df.groupby("experiment_id_first")["recall_1_mean"].idxmax()]
    df_recall["experiment_id_first"] = df_recall["experiment_id_first"].astype(int)
    df_recall.sort_values(by="experiment_id_first", inplace=True)
    df_recall["experiment_idx"] = np.arange(1, len(df_recall) + 1)

    _create_formatted_values(df_recall, "recall_1")

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
            y=alt.Y("recall_1_mean:Q", title="Recall Score"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx:N", title="Experiments"),
                alt.Tooltip("parent_run_name_first:N", title="Models"),
                alt.Tooltip("formatted_values:N", title="Recall Score"),
            ],
        )
        + (
            alt.Chart(data=df_recall)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("recall_1_mean:Q", title="Recall Score"),
                yError=alt.YError("recall_1_std"),
            )
        ),
        use_container_width=True,
        theme=None,
    )

    st.dataframe(df_recall)


def plot_worst_by_recall_score(df: pd.DataFrame):
    df_recall_min = df.loc[df.groupby("experiment_id_first")["recall_1_mean"].idxmin()]
    df_recall_min["experiment_id_first"] = df_recall_min["experiment_id_first"].astype(int)  # fmt: skip
    df_recall_min.sort_values(by="experiment_id_first", inplace=True)
    df_recall_min["experiment_idx"] = np.arange(1, len(df_recall_min) + 1)

    _create_formatted_values(df_recall_min, "recall_1")

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
            y=alt.Y("recall_1_mean:Q", title="Recall Score"),
            color=alt.Color("parent_run_name_first:N", title="Models"),
            tooltip=[
                alt.Tooltip("experiment_idx:N", title="Experiments"),
                alt.Tooltip("parent_run_name_first:N", title="Models"),
                alt.Tooltip("formatted_values:N", title="Recall Score"),
            ],
        )
        + (
            alt.Chart(data=df_recall_min)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("recall_1_mean:Q", title="Recall Score"),
                yError=alt.YError("recall_1_std"),
            )
        ),
        use_container_width=True,
        theme=None,
    )

    st.dataframe(df_recall_min)


def plot_recall_score_by_experiments(df_cv_metrics: pd.DataFrame):
    _create_formatted_values(df_cv_metrics, "recall_1")

    return st.altair_chart(
        alt.Chart(
            data=df_cv_metrics,
            title=alt.TitleParams(
                text="Recall Score Across Experiments",
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
            y=alt.Y("recall_1_mean:Q", title="Recall"),
            tooltip=[
                alt.Tooltip("experiment_idx", title="Experiments"),
                alt.Tooltip("formatted_values", title="Recall Score"),
            ],
        )
        + (
            alt.Chart(df_cv_metrics)
            .mark_errorbar()
            .encode(
                x=alt.X("experiment_idx:N", title="Experiments"),
                y=alt.Y("recall_1_mean:Q", title="Recall Score"),
                yError=alt.YError("recall_1_std:Q"),
            )
        ),
        use_container_width=True,
        theme=None,
    )
