import mlflow
import streamlit as st
from crunchy_mining import mlflow_util
from crunchy_mining.util import summarize_classification
from crunchy_mining.util import summarize_resource

from pages.fragments import plot_best_by_recall_score
from pages.fragments import plot_worst_by_recall_score

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5002")

with st.spinner("Fetching experiment data..."):
    df = mlflow_util.get_cv_metrics_by_task(task_name="clf/")

outputs = summarize_classification(df)
outputs_res = summarize_resource(df)
chart_conf = {"use_container_width": True, "theme": None}

plot_best_by_recall_score(df)
plot_worst_by_recall_score(df)

st.altair_chart(outputs["auc_chart"], **chart_conf)
st.dataframe(outputs["auc"], use_container_width=True)

st.altair_chart(outputs["auc_min_chart"], **chart_conf)
st.dataframe(outputs["auc_min"], use_container_width=True)

st.altair_chart(outputs_res["fit_time_chart"], **chart_conf)
st.dataframe(outputs_res["fit_time"], use_container_width=True)

st.altair_chart(outputs_res["fit_time_max_chart"], **chart_conf)
st.dataframe(outputs_res["fit_time_max"], use_container_width=True)

st.altair_chart(outputs_res["score_time_chart"], **chart_conf)
st.dataframe(outputs_res["score_time"], use_container_width=True)

st.altair_chart(outputs_res["score_time_max_chart"], **chart_conf)
st.dataframe(outputs_res["score_time_max"], use_container_width=True)

st.altair_chart(outputs_res["fit_memory_chart"], **chart_conf)
st.dataframe(outputs_res["fit_memory"], use_container_width=True)

st.altair_chart(outputs_res["fit_memory_max_chart"], **chart_conf)
st.dataframe(outputs_res["fit_memory_max"], use_container_width=True)

st.altair_chart(outputs_res["score_memory_chart"], **chart_conf)
st.dataframe(outputs_res["score_memory"], use_container_width=True)

st.altair_chart(outputs_res["score_memory_max_chart"], **chart_conf)
st.dataframe(outputs_res["score_memory_max"], use_container_width=True)
