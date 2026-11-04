
import os

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# Setup
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# Task 1: Load input data and model artifacts
# --------------------------------------------------

new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

model = joblib.load("models/delay_model.pkl")

threshold = float(
    pd.read_csv("models/delay_threshold.csv")
    .loc[0, "threshold"]
)

feature_order = (
    pd.read_csv("models/feature_order.csv")["feature"]
    .tolist()
)

impute_df = pd.read_csv("models/impute_values.csv")

medians = dict(
    zip(impute_df["column"], impute_df["median"])
)

hub_segments = pd.read_csv("outputs/hub_segments.csv")

print("New shipments:", len(new_shipments))
print("Threshold:", threshold)


# --------------------------------------------------
# Task 2: Apply historical preprocessing
# --------------------------------------------------

# Reuse historical medians. Do not calculate new medians.
cleaned_shipments, _ = clean_shipments(
    new_shipments,
    medians=medians
)

featured_shipments = build_features(
    cleaned_shipments,
    hubs
)

# Preserve original input order.
featured_shipments["_input_order"] = np.arange(
    len(featured_shipments)
)

X_new = featured_shipments[feature_order].copy()

assert X_new.columns.tolist() == feature_order
assert not X_new.isnull().any().any()


# --------------------------------------------------
# Task 3: Predict delay probabilities
# --------------------------------------------------

probabilities = model.predict_proba(X_new)[:, 1]

# Round probabilities for the output file.
rounded_probabilities = np.round(probabilities, 4)

flagged = (
    rounded_probabilities >= threshold
).astype(int)

featured_shipments["delay_probability"] = rounded_probabilities
featured_shipments["flagged"] = flagged


# --------------------------------------------------
# Task 4: Attach hub segments
# --------------------------------------------------

featured_shipments = featured_shipments.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    sort=False,
    validate="many_to_one"
)

featured_shipments = (
    featured_shipments
    .sort_values("_input_order")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Task 5: Generate operational actions
# --------------------------------------------------

featured_shipments["action"] = np.select(
    [
        (
            (featured_shipments["flagged"] == 1)
            & (featured_shipments["segment"] == "Escalate")
        ),
        featured_shipments["flagged"] == 1
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call"
    ],
    default="No action"
)


# --------------------------------------------------
# Task 6: Save final output
# --------------------------------------------------

output_columns = [
    "shipment_id",
    "hub_id",
    "segment",
    "delay_probability",
    "flagged",
    "action"
]

output_df = featured_shipments[output_columns].copy()

# Acceptance checks.
assert len(output_df) == 48
assert output_df["shipment_id"].is_unique
assert not output_df.isnull().any().any()

assert (
    output_df["flagged"]
    == (output_df["delay_probability"] >= threshold).astype(int)
).all()

assert (
    output_df.loc[
        output_df["flagged"] == 0, "action"
    ] == "No action"
).all()

assert (
    output_df.loc[
        (output_df["flagged"] == 1)
        & (output_df["segment"] == "Escalate"),
        "action"
    ] == "Escalate to hub manager"
).all()

assert (
    output_df.loc[
        (output_df["flagged"] == 1)
        & (output_df["segment"] != "Escalate"),
        "action"
    ] == "Proactive customer call"
).all()

output_df.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nNew Shipment Prediction Summary")
print("Total shipments:", len(output_df))
print("Flagged shipments:", int(output_df["flagged"].sum()))
print(
    "Escalations:",
    int(
        (
            output_df["action"]
            == "Escalate to hub manager"
        ).sum()
    )
)
print(
    "Proactive calls:",
    int(
        (
            output_df["action"]
            == "Proactive customer call"
        ).sum()
    )
)
print(
    "No action:",
    int(
        (
            output_df["action"]
            == "No action"
        ).sum()
    )
)

print("\nAction Distribution")
print(output_df["action"].value_counts())

print("\nOutput Preview")
print(output_df.to_string(index=False))

print("\nPredictions saved to outputs/new_shipment_actions.csv")
