import mlflow
import streamlit as st
from crunchy_mining import mlflow_util
from crunchy_mining.util import plot_confusion_matrix
from crunchy_mining.util import plot_evaluation_stability
from crunchy_mining.util import plot_resource_stability
from crunchy_mining.util import tabulate_classification_report
from crunchy_mining.util import tabulate_resource_usage
from mlflow import MlflowClient

from pages.fragments import create_experiment_model_selector
from pages.fragments import create_experiment_selector
from pages.fragments import create_fold_selector
from pages.fragments import plot_mean_roc_curve
from pages.fragments import plot_roc_curve

st.set_page_config(layout="wide")

tabs = st.tabs(["Experiments", "Model-Specific", "Fold-Specific"])

with tabs[0]:
    experiment = create_experiment_selector()
    df_val, df_cv = mlflow_util.get_val_cv_metrics_by_experiment(experiment)

    st.markdown("**Validation**")
    st.dataframe(df_val)

    st.markdown("**Cross-Validation**")
    st.dataframe(df_cv)

with tabs[1]:
    experiment, model = create_experiment_model_selector()
    mlflow.set_experiment(experiment)

    with st.spinner("Fetching experiment data..."):
        parent_run_id = mlflow_util.get_latest_run_id_by_name(model)

        validation_run = mlflow_util.get_nested_runs_by_parent_id(
            parent_run_id,
            filter_string="run_name = 'validation'",
        )

        cv_runs = mlflow_util.get_nested_runs_by_parent_id(
            parent_run_id,
            filter_string="run_name LIKE 'fold%'",
        )

        if validation_run is None or cv_runs is None:
            run_id_warn = "Model training is required. Please train the model before proceeding."  # fmt: skip
            st.warning(run_id_warn)
            st.stop()

        val_metrics = (
            validation_run.filter(like="metrics", axis=1)
            .iloc[0]
            .rename(lambda x: x.replace("metrics.", ""))
            .to_dict()
        )

        cv_metrics = (
            cv_runs.filter(like="metrics", axis=1)
            .mean()
            .rename(lambda x: x.replace("metrics.", ""))
            .to_dict()
        )

    st.markdown("**Validation**")
    cols = st.columns([1.25, 1])

    cols[0].markdown("**Classification Report**")
    cols[0].table(tabulate_classification_report(val_metrics))

    artifact_uri = f"runs:/{validation_run.run_id[0]}/roc.json"
    roc = mlflow.artifacts.load_dict(artifact_uri)
    cols[0].altair_chart(plot_roc_curve(roc), use_container_width=True, theme=None)

    cols[0].markdown("**Resource Usage**")
    cols[0].table(tabulate_resource_usage(val_metrics))

    cols[1].altair_chart(
        plot_confusion_matrix(val_metrics),
        use_container_width=True,
    )

    st.divider()
    st.markdown("**Cross-Validation**")
    cols = st.columns([1.25, 1])

    cols[0].markdown("**Classification Report**")
    cols[0].table(tabulate_classification_report(cv_metrics))

    rocs = []
    for run in cv_runs.itertuples():
        artifact_uri = f"runs:/{run.run_id}/roc.json"
        roc = mlflow.artifacts.load_dict(artifact_uri)
        rocs.append(roc)

    cols[0].altair_chart(
        plot_mean_roc_curve(rocs),
        use_container_width=True,
        theme=None,
    )

    cols[0].markdown("**Resource Usage**")
    cols[0].table(tabulate_resource_usage(cv_metrics))

    cols[1].altair_chart(
        plot_confusion_matrix(cv_metrics),
        use_container_width=True,
    )

    cv_folds_eval = (
        cv_runs[
            [
                "tags.mlflow.runName",
                "metrics.false_positive_rate",
                "metrics.precision_1",
                "metrics.recall_1",
                "metrics.roc_auc",
            ]
        ]
        .rename({"tags.mlflow.runName": "folds"}, axis=1)
        .rename(lambda x: x.replace("metrics.", ""), axis=1)
        .melt(id_vars="folds", var_name="metrics", value_name="value")
    )

    cv_folds_time = (
        cv_runs[
            [
                "tags.mlflow.runName",
                "metrics.fit_time",
                "metrics.score_time",
            ]
        ]
        .rename({"tags.mlflow.runName": "folds"}, axis=1)
        .rename(lambda x: x.replace("metrics.", ""), axis=1)
        .melt(id_vars="folds", var_name="metrics", value_name="value")
        .assign(value=lambda x: x["value"] / 1_000_000)
    )

    cv_folds_memory = (
        cv_runs[
            [
                "tags.mlflow.runName",
                "metrics.fit_memory_peak",
                "metrics.score_memory_peak",
            ]
        ]
        .rename({"tags.mlflow.runName": "folds"}, axis=1)
        .rename(lambda x: x.replace("metrics.", ""), axis=1)
        .rename(lambda x: x.replace("_peak", ""), axis=1)
        .melt(id_vars="folds", var_name="metrics", value_name="value")
        .assign(value=lambda x: x["value"] / 1_000_000)
    )

    st.divider()
    eval_stb_chart = plot_evaluation_stability(cv_folds_eval)
    st.altair_chart(eval_stb_chart, use_container_width=True, theme=None)

    res_stb_charts = plot_resource_stability(cv_folds_time, cv_folds_memory)
    cols = st.columns([1, 1])
    cols[0].altair_chart(res_stb_charts[0], use_container_width=True, theme=None)
    cols[1].altair_chart(res_stb_charts[1], use_container_width=True, theme=None)

with tabs[2]:
    experiment, model, fold = create_fold_selector()
    mlflow.set_experiment(experiment)

    with st.spinner("Fetching experiment data..."):
        parent_run_id = mlflow_util.get_latest_run_id_by_name(model)
        run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=fold)

    if not run_id:
        run_id_warn = "Model training is required. Please train the model before proceeding."  # fmt: skip
        st.warning(run_id_warn)
        st.stop()

    with st.spinner("Fetching run metrics..."):
        client = MlflowClient()
        run = client.get_run(run_id)
        metrics = run.data.metrics

    cols = st.columns([1.25, 1])

    cols[0].markdown("**Classification Report**")
    cols[0].table(tabulate_classification_report(metrics))

    artifact_uri = f"runs:/{run_id}/roc.json"
    roc = mlflow.artifacts.load_dict(artifact_uri)
    cols[0].altair_chart(plot_roc_curve(roc), use_container_width=True, theme=None)

    cols[0].markdown("**Resource Usage**")
    cols[0].table(tabulate_resource_usage(metrics))

    cols[1].altair_chart(plot_confusion_matrix(metrics), use_container_width=True)
