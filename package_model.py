
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# --------------------------------------------------
# Setup
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# Task 1: Load final configuration
# --------------------------------------------------

config_df = pd.read_csv("outputs/part_c_final_config.csv")

assert len(config_df) == 1, "Expected exactly one final configuration row."

config = config_df.iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])


def parse_max_depth(value):
    if pd.isna(value):
        return None

    if str(value).strip().lower() in {
        "unlimited", "none", "nan", ""
    }:
        return None

    return int(float(value))


max_depth = parse_max_depth(config["max_depth"])

print("Final Model Configuration")
print("Selector k:", selector_k)
print("Estimators:", n_estimators)
print("Max depth:", max_depth)
print("Minimum samples per leaf:", min_samples_leaf)
print("Threshold:", threshold)


# --------------------------------------------------
# Task 2: Rebuild Part C stratified split
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES].copy()
y = df["is_delayed"].copy()

assert X.columns.tolist() == FEATURES
assert len(FEATURES) == 17
assert not X.isnull().any().any()
assert not y.isnull().any()

# First split: 70% training, 30% temporary.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# Second split: divide temporary data equally.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\nRebuilt Dataset Split")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

assert len(X_train) == 840
assert len(X_val) == 180
assert len(X_test) == 180

assert int(y_train.sum()) == 195
assert int(y_val.sum()) == 42
assert int(y_test.sum()) == 41


# --------------------------------------------------
# Task 3: Rebuild and fit final pipeline
# --------------------------------------------------

final_pipeline = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE
        )
    )
])

# Fit on the original 17 features.
# The pipeline performs feature selection internally.
final_pipeline.fit(X_train, y_train)

print("\nFinal pipeline fitted.")


# --------------------------------------------------
# Task 4: Obtain historical imputation medians
# --------------------------------------------------

raw_df = pd.read_csv("data/raw_shipments.csv")

# clean_shipments returns (clean_df, medians).
# No medians argument means calculate from historical data.
cleaned_historical_df, medians = clean_shipments(raw_df)

expected_median_keys = [
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr"
]

assert list(medians.keys()) == expected_median_keys

impute_values_df = pd.DataFrame([
    {
        "column": column,
        "median": float(medians[column])
    }
    for column in expected_median_keys
])

impute_values_df.to_csv(
    "models/impute_values.csv",
    index=False,
    float_format="%.4f"
)

print("\nHistorical Imputation Values")
print(impute_values_df.to_string(index=False))


# --------------------------------------------------
# Task 5: Save model and supporting files
# --------------------------------------------------

# Save fitted pipeline.
joblib.dump(
    final_pipeline,
    "models/delay_model.pkl"
)

# Save decision threshold.
pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    "models/delay_threshold.csv",
    index=False,
    float_format="%.4f"
)

# Save exact model feature order.
pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    "models/feature_order.csv",
    index=False
)

print("\nModel artifacts saved:")
print("models/delay_model.pkl")
print("models/delay_threshold.csv")
print("models/feature_order.csv")
print("models/impute_values.csv")


# --------------------------------------------------
# Task 6: Reload model and verify probabilities
# --------------------------------------------------

reloaded_pipeline = joblib.load(
    "models/delay_model.pkl"
)

# Predict test probabilities using both pipelines.
original_probabilities = (
    final_pipeline.predict_proba(X_test)[:, 1]
)

reloaded_probabilities = (
    reloaded_pipeline.predict_proba(X_test)[:, 1]
)

reload_max_abs_probability_diff = float(
    np.max(
        np.abs(
            original_probabilities - reloaded_probabilities
        )
    )
)

reload_results = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": reload_max_abs_probability_diff
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_results.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False,
    float_format="%.4f"
)

print("\nReload Verification")
print(reload_results.to_string(index=False))

assert len(X_test) == 180
assert reload_max_abs_probability_diff <= 1e-9

print("\nModel packaging completed successfully.")
