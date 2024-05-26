import time
import warnings

import mlflow
import numpy as np
from crunchy_mining import mlflow_util
from crunchy_mining.preprocessing.preprocessors import PreprocessorV3
from crunchy_mining.util import custom_predict
from dotenv import load_dotenv
from hydra import compose
from hydra import initialize

warnings.filterwarnings(
    action="ignore",
    message=".*X does not have valid feature name.*",
)

load_dotenv()

with initialize(version_base=None, config_path="./conf"):
    cfg = compose(overrides=["+experiment=retrain_v9"])

feature_names = cfg.vars.categorical + cfg.vars.numerical

v_not_fraud = {
    "income": 0.5,
    "name_email_similarity": 0.5,
    "prev_address_months_count": 12,
    "current_address_months_count": 12,
    "customer_age": 18,
    "days_since_request": 7,
    "intended_balcon_amount": 50,
    "payment_type": "AA",
    "zip_count_4w": 100,
    "velocity_6h": 100,
    "velocity_24h": 1000,
    "velocity_4w": 2000,
    "bank_branch_count_8w": 500,
    "date_of_birth_distinct_emails_4w": 10,
    "employment_status": "CA",
    "credit_risk_score": 50,
    "email_is_free": 1,
    "housing_status": "BA",
    "phone_home_valid": 1,
    "phone_mobile_valid": 1,
    "bank_months_count": 12,
    "has_other_cards": 1,
    "proposed_credit_limit": 1000,
    "foreign_request": 1,
    "source": "INTERNET",
    "session_length_in_minutes": 30,
    "device_os": "windows",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 0,
    "device_fraud_count": 0,
    "month": 6,
}

v_fraud = {
    "income": 0.5,
    "name_email_similarity": 0.5,
    "prev_address_months_count": 0,
    "current_address_months_count": 12,
    "customer_age": 70,
    "days_since_request": 7,
    "intended_balcon_amount": 50,
    "payment_type": "AA",
    "zip_count_4w": 100,
    "velocity_6h": 100,
    "velocity_24h": 1000,
    "velocity_4w": 2000,
    "bank_branch_count_8w": 500,
    "date_of_birth_distinct_emails_4w": 10,
    "employment_status": "CA",
    "credit_risk_score": 50,
    "email_is_free": 1,
    "housing_status": "BA",
    "phone_home_valid": 1,
    "phone_mobile_valid": 1,
    "bank_months_count": 12,
    "has_other_cards": 1,
    "proposed_credit_limit": 1000,
    "foreign_request": 1,
    "source": "INTERNET",
    "session_length_in_minutes": 30,
    "device_os": "windows",
    "keep_alive_session": 1,
    "device_distinct_emails_8w": 0,
    "device_fraud_count": 0,
    "month": 6,
}


mlflow.set_experiment("clf_retrain/retrain_v9")
preprocessor = PreprocessorV3(cfg)
preprocessor.load_encoders()

parent_run_id = mlflow_util.get_latest_run_id_by_name("LightGBM")
run = mlflow_util.get_nested_runs_by_parent_id(parent_run_id, "run_name = 'testing'")
threshold = run.iloc[0]["metrics.threshold"]
run_id = run.iloc[0]["run_id"]

model_uri = f"runs:/{run_id}/model"
model = mlflow.lightgbm.load_model(model_uri)

n_inferences = 10_000
start = time.perf_counter_ns()

for i in range(n_inferences):
    if i % 2 == 0:
        v_ordered = [v_fraud[feature] for feature in feature_names]
    else:
        v_ordered = [v_not_fraud[feature] for feature in feature_names]

    v_ft = np.array(v_ordered).reshape(1, -1)
    v_ft_pp = preprocessor.transform(v_ft)

    y_prob = model.predict_proba(v_ft_pp.astype(np.float64))
    y = custom_predict(y_prob=y_prob, threshold=threshold)[0]

end = time.perf_counter_ns()
duration = end - start

print(f"Inferences: {n_inferences}")
print(f"Duration: {duration / 1_000_000} milliseconds")
print(f"Duration (1): {duration / 1_000_000 / n_inferences} milliseconds")
print(f"Inferences/s: {1 / (duration / 1_000_000_000 / n_inferences)}")
