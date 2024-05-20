import mlflow
import streamlit as st
from crunchy_mining import mlflow_util
from crunchy_mining.util import plot_intrinsic_importances

from pages.fragments import create_fold_selector

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5002")

experiment, model, fold = create_fold_selector()

mlflow.set_experiment(experiment)

parent_run_id = mlflow_util.get_latest_run_id_by_name(model)
run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name=fold)

if not run_id:
    st.warning("Model training is required. Please train the model before proceeding.")
    st.stop()

st.markdown("**Intrinsic and Model Specific**")

artifact_uri = f"runs:/{run_id}/interpretation/intrinsic.json"
importances = mlflow_util.load_table(artifact_uri)

if importances is None:
    st.warning("Hit a roadblock! Consider running the function to generate feature importance.")  # fmt: skip
    st.stop()

chart = plot_intrinsic_importances(importances, name=model)
cols = st.columns([1, 1])
cols[0].altair_chart(chart, use_container_width=True, theme=None)
cols[1].dataframe(
    importances.sort_values(by="importances", ascending=False).set_index(
        keys="feature_names"
    ),
    use_container_width=True,
)
