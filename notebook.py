# %%
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# %%
df = pd.read_csv("data/Base.csv")

# %%
df.info()

# %%
df["fraud_bool"].value_counts()

# %%
counts = df["fraud_bool"].value_counts()
majority_class = counts.index[np.argmax(counts)]
minority_class = counts.index[np.argmin(counts)]
n_minority_class = np.min(counts)

df_sampled = pd.concat(
    [
        df[df["fraud_bool"] == majority_class].sample(
            n=n_minority_class,
            random_state=12345,
        ),
        df[df["fraud_bool"] == minority_class],
    ]
)

# %%
variables = {
    "categorical": [
        "payment_type",
        "employment_status",
        "email_is_free",
        "housing_status",
        "phone_home_valid",
        "phone_mobile_valid",
        "has_other_cards",
        "foreign_request",
        "source",
        "device_os",
        "keep_alive_session",
    ],
    "numerical": [
        "income",
        "name_email_similarity",
        "prev_address_months_count",
        "current_address_months_count",
        "customer_age",
        "days_since_request",
        "intended_balcon_amount",
        "zip_count_4w",
        "velocity_6h",
        "velocity_24h",
        "velocity_4w",
        "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w",
        "credit_risk_score",
        "bank_months_count",
        "proposed_credit_limit",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
        "device_fraud_count",
        "month",
    ],
    "target": "fraud_bool",
}

# %%
df_train, df_test = train_test_split(
    df_sampled,
    test_size=0.15,
    random_state=12345,
    stratify=df_sampled[variables["target"]],
)

# %%
pd.concat(
    [
        df_train[variables["target"]].value_counts(),
        df_test[variables["target"]].value_counts(),
    ]
)

# %%
df_train_sm, df_val = train_test_split(
    df_train,
    test_size=0.15 / 0.85,
    random_state=12345,
    stratify=df_train[variables["target"]],
)

# %%
pd.concat(
    [
        df_train_sm[variables["target"]].value_counts(),
        df_val[variables["target"]].value_counts(),
    ]
)

# %%
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_sm_cat = oe.fit_transform(df_train_sm[variables["categorical"]])
X_val_cat = oe.transform(df_val[variables["categorical"]])
X_test_cat = oe.transform(df_test[variables["categorical"]])

# %%
mm = MinMaxScaler()
X_train_sm_cat_scaled = mm.fit_transform(X_train_sm_cat)
X_val_cat_scaled = mm.transform(X_val_cat)
X_test_cat_scaled = mm.transform(X_test_cat)

# %%
ss = StandardScaler()
X_train_sm_num = ss.fit_transform(df_train_sm[variables["numerical"]])
X_val_num = ss.transform(df_val[variables["numerical"]])
X_test_num = ss.transform(df_test[variables["numerical"]])

# %%
X_train_sm_ft = np.hstack((X_train_sm_cat_scaled, X_train_sm_num))
X_val_ft = np.hstack((X_val_cat_scaled, X_val_num))
X_test_ft = np.hstack((X_test_cat_scaled, X_test_num))

# %%
y_train_sm = df_train_sm[variables["target"]].to_numpy()
y_val = df_val[variables["target"]].to_numpy()
y_test = df_test[variables["target"]].to_numpy()

# %%
knn = KNeighborsClassifier()
knn.fit(X_train_sm_ft, y_train_sm)

# %%
logreg = LogisticRegression(random_state=12345, n_jobs=-1)
logreg.fit(X_train_sm_ft, y_train_sm)

# %%
gnb = GaussianNB()
gnb.fit(X_train_sm_ft, y_train_sm)

# %%
svc = LinearSVC(dual="auto", random_state=12345)
svc.fit(X_train_sm_ft, y_train_sm)

# %%
dt = DecisionTreeClassifier(random_state=12345)
dt.fit(X_train_sm_ft, y_train_sm)

# %%
ab = AdaBoostClassifier(algorithm="SAMME", random_state=12345)
ab.fit(X_train_sm_ft, y_train_sm)

# %%
rf = RandomForestClassifier(n_jobs=-1, random_state=12345)
rf.fit(X_train_sm_ft, y_train_sm)

# %%
xgb = XGBClassifier(n_jobs=-1, random_state=12345)
xgb.fit(X_train_sm_ft, y_train_sm)

# %%
lgb = LGBMClassifier(random_state=12345, n_jobs=-1)
lgb.fit(X_train_sm_ft, y_train_sm)

# %%
catb = CatBoostClassifier(metric_period=250, random_state=12345)
catb.fit(X_train_sm_ft, y_train_sm)

# %%
y_knn = knn.predict(X_val_ft)
print(classification_report(y_val, y_knn))

# %%
y_logreg = logreg.predict(X_val_ft)
print(classification_report(y_val, y_logreg))

# %%
y_gnb = gnb.predict(X_val_ft)
print(classification_report(y_val, y_gnb))

# %%
y_svc = svc.predict(X_val_ft)
print(classification_report(y_val, y_svc))

# %%
y_dt = dt.predict(X_val_ft)
print(classification_report(y_val, y_dt))

# %%
y_ab = ab.predict(X_val_ft)
print(classification_report(y_val, y_ab))

# %%
y_rf = rf.predict(X_val_ft)
print(classification_report(y_val, y_rf))

# %%
y_xgb = xgb.predict(X_val_ft)
print(classification_report(y_val, y_xgb))

# %%
y_lgb = lgb.predict(X_val_ft)
print(classification_report(y_val, y_lgb))

# %%
y_catb = catb.predict(X_val_ft)
print(classification_report(y_val, y_catb))

# %%
