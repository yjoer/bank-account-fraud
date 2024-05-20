import mlflow
import numpy as np
import streamlit as st
from crunchy_mining.mlflow_util import get_cv_metrics_by_model
from crunchy_mining.pages.fragments import plot_auc_score_by_experiments
from crunchy_mining.pages.fragments import plot_memory_by_experiments
from crunchy_mining.pages.fragments import plot_time_by_experiments

from pages.fragments import create_model_selector
from pages.fragments import plot_recall_score_by_experiments

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5002")

task = "clf"
model = create_model_selector()

with st.spinner("Fetching experiment data..."):
    df_cv_metrics = get_cv_metrics_by_model(task, model)
    df_cv_metrics["experiment_id_first"] = df_cv_metrics["experiment_id_first"].astype(int)  # fmt: skip
    df_cv_metrics.sort_values(by="experiment_id_first", inplace=True)
    df_cv_metrics["experiment_idx"] = np.arange(1, len(df_cv_metrics) + 1)

plot_recall_score_by_experiments(df_cv_metrics)
plot_auc_score_by_experiments(df_cv_metrics)

plot_time_by_experiments(df_cv_metrics)
plot_memory_by_experiments(df_cv_metrics)

st.dataframe(df_cv_metrics)
