import hydra
import mlflow
import numpy as np
import streamlit as st
from crunchy_mining import mlflow_util
from crunchy_mining.preprocessing.preprocessors import PreprocessorV3
from crunchy_mining.preprocessing.preprocessors import PreprocessorV5
from crunchy_mining.preprocessing.preprocessors import PreprocessorV6
from crunchy_mining.preprocessing.preprocessors import PreprocessorV7
from crunchy_mining.util import custom_predict
from hydra import compose
from hydra import initialize

from pages.fragments import create_model_selector

st.set_page_config(layout="wide")

st.markdown("**1. Select a Model**")
model_name = create_model_selector()

model_experiment_map = {
    "KNN": "clf_retrain/retrain_v1",
    "Logistic Regression": "clf_retrain/retrain_v2",
    "Gaussian NB": "clf_retrain/retrain_v3",
    "Linear SVC": "clf_retrain/retrain_v4",
    "Decision Tree": "clf_retrain/retrain_v5",
    "Random Forest": "clf_retrain/retrain_v6",
    "AdaBoost": "clf_retrain/retrain_v7",
    "XGBoost": "clf_retrain/retrain_v8",
    "LightGBM": "clf_retrain/retrain_v9",
    "CatBoost": "clf_retrain/retrain_v10",
}

experiment = model_experiment_map[model_name]
_, experiment_file = experiment.split("/")[:2]

# Allow fast consecutive re-runs without crashing the app.
hydra.core.global_hydra.GlobalHydra.instance().clear()

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(overrides=[f"+experiment={experiment_file}"])


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str):
    parent_run_id = mlflow_util.get_latest_run_id_by_name(model_name)
    run_id = mlflow_util.get_nested_run_ids_by_parent_id(parent_run_id, name="testing")

    model_uri = f"runs:/{run_id}/model"

    match model_name:
        case "XGBoost":
            model = mlflow.xgboost.load_model(model_uri)
        case "LightGBM":
            model = mlflow.lightgbm.load_model(model_uri)
        case "CatBoost":
            model = mlflow.catboost.load_model(model_uri)
        case _:
            model = mlflow.sklearn.load_model(model_uri)

    return parent_run_id, run_id, model


mlflow.set_experiment(experiment)
parent_run_id, run_id, model = load_model(model_name)
st.success(f"{model_name} loaded successfully!")


@st.cache_resource(show_spinner="Loading encoders and scalers...")
def load_preprocessor(variant: int):
    match variant:
        case 3:
            preprocessor = PreprocessorV3(cfg)
        case 5:
            preprocessor = PreprocessorV5(cfg)
        case 6:
            preprocessor = PreprocessorV6(cfg)
        case 7:
            preprocessor = PreprocessorV7(cfg)

    preprocessor.load_encoders()

    return preprocessor


pp_variant = cfg.preprocessing.variant
preprocessor = load_preprocessor(pp_variant)
st.success(f"{preprocessor.__class__.__name__} loaded successfully!")

st.markdown("**2. Specify the Inputs (31)**")

if st.toggle("Simulate Fraudulent Applications"):
    st.session_state["prev_address_months_count"] = 0
    st.session_state["customer_age"] = 70
else:
    st.session_state["prev_address_months_count"] = 12
    st.session_state["customer_age"] = 18

v = {}

cols = st.columns([1, 1, 1])
v["income"] = cols[0].number_input("Income [0.1, 0.9]", 0.1, 0.9, 0.5, 0.01)
v["name_email_similarity"] = cols[1].number_input("Name Email Similarity [0, 1]", 0.0, 1.0, 0.5, 0.01)  # fmt: skip
v["prev_address_months_count"] = cols[2].number_input("Previous Address Months [-1, 380]", -1, None, st.session_state["prev_address_months_count"], 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["current_address_months_count"] = cols[0].number_input("Current Address Months [-1, 429]", -1, None, 12, 1)  # fmt: skip
v["customer_age"] = cols[1].number_input("Customer Age [10, 90]", 1, 100,  st.session_state["customer_age"], 1)  # fmt: skip
v["days_since_request"] = cols[2].number_input("Days Since Request [0, 79]", 0, None, 7, 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["intended_balcon_amount"] = cols[0].number_input("Intended Balcon Amount [-1, 114]", 0, None, 50, 1)  # fmt: skip
v["payment_type"] = cols[1].selectbox("Payment Type", ["AA", "AB", "AC", "AD", "AE"])
v["zip_count_4w"] = cols[2].number_input("Same Zip Code Applications [1, 6830]", 0, None, 100, 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["velocity_6h"] = cols[0].number_input("Velocity - 6 hours [0, 16818]", 0, None, 100, 1)  # fmt: skip
v["velocity_24h"] = cols[1].number_input("Velocity - 1 day [1297, 9586]", 0, None, 1000, 1)  # fmt: skip
v["velocity_4w"] = cols[2].number_input("Velocity - 1 month [2825, 7020]", 0, None, 2000, 1)  # fmt: skip

cols = st.columns([1, 1])
v["bank_branch_count_8w"] = cols[0].number_input("Bank Branch Applications - 1 month [0, 2404]", 0, None, 500, 1)  # fmt: skip
v["date_of_birth_distinct_emails_4w"] = cols[1].number_input("Distinct Emails With the Same DOB - 1 month [0, 39]", 0, None, 10, 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["employment_status"] = cols[0].selectbox("Employment Status", ["CA", "CB", "CC", "CD", "CE", "CF", "CG"])  # fmt: skip
v["credit_risk_score"] = cols[1].number_input("Credit Risk Score [-191, 389]", None, None, 50, 1)  # fmt: skip
v["email_is_free"] = cols[2].number_input("Free Email [0, 1]", 0, 1, 1, 1)

cols = st.columns([1, 1, 1])
v["housing_status"] = cols[0].selectbox("Housing Status", ["BA", "BB", "BC", "BD", "BE", "BF", "BG"])  # fmt: skip
v["phone_home_valid"] = cols[1].number_input("Home Phone Number Valid [0, 1]", 0, 1, 1, 1)  # fmt: skip
v["phone_mobile_valid"] = cols[2].number_input("Mobile Phone Number Valid [0, 1]", 0, 1, 1, 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["bank_months_count"] = cols[0].number_input("Previous Bank Account Months [-1, 32]", -1, None, 12, 1)  # fmt: skip
v["has_other_cards"] = cols[1].number_input("Other Cards [0, 1]", 0, 1, 1, 1)
v["proposed_credit_limit"] = cols[2].number_input("Proposed Credit Limit [200, 2000]", 0, None, 1000, 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["foreign_request"] = cols[0].number_input("Foreign Request [0, 1]", 0, 1, 1, 1)
v["source"] = cols[1].selectbox("Source", ["INTERNET", "TELEAPP"])
v["session_length_in_minutes"] = cols[2].number_input("Session Length in Minutes [-1, 107]", -1, None, 30, 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["device_os"] = cols[0].selectbox("Device OS", ["windows", "macintosh", "linux", "x11", "other"])  # fmt: skip
v["keep_alive_session"] = cols[1].number_input("Keep Alive Session [0, 1]", 0, 1, 1, 1)
v["device_distinct_emails_8w"] = cols[2].number_input("Device Distinct Emails - 2 months [-1, 2]", -1, None, 0, 1)  # fmt: skip

cols = st.columns([1, 1, 1])
v["device_fraud_count"] = cols[0].number_input("Device Fraud Count [0, 1]", 0, None, 0, 1)  # fmt: skip
v["month"] = cols[1].number_input("Application Month [0, 7]", 0, 12, 6, 1)  # fmt: skip

st.markdown("**3. Get the Prediction**")

feature_names = cfg.vars.categorical + cfg.vars.numerical
v_ordered = [v[feature] for feature in feature_names]
v_ft = np.array(v_ordered).reshape(1, -1)
v_ft_pp = preprocessor.transform(v_ft)

run = mlflow_util.get_nested_runs_by_parent_id(parent_run_id, "run_name = 'testing'")
threshold = run.iloc[0]["metrics.threshold"]
false_positive_rate = run.iloc[0]["metrics.false_positive_rate"]

y_prob = model.predict_proba(v_ft_pp.astype(np.float64))
y = custom_predict(y_prob=y_prob, threshold=threshold)[0]

st.text(f"Prediction Probability: {y_prob}")
st.text(f"Threshold: {threshold}")
st.text(f"False Positive Rate: {false_positive_rate}")

if y == 0:
    st.success("Not Fraud")
elif y == 1:
    st.error("Fraud")
