import os
import pandas as pd
import numpy as np
from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES,
    PROMISED_HOURS,
)

os.makedirs("outputs", exist_ok=True)

raw= pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

cleaned, medians = clean_shipments(raw)
features = build_features(cleaned, hubs)

features['is_delayed'] = (features['delivery_hours'] > features['promised_hours']).astype(int)

raw_rows = len(raw)
raw_dedup= raw.drop_duplicates().copy().reset_index(drop=True)
raw_dedup_rows = len(raw_dedup)
duplicate_rows_removed = raw_rows - raw_dedup_rows

# Missing Values Before

missing_distance_before = int(raw_dedup["distance_km"].isna().sum())
missing_weight_before = int(raw_dedup['parcel_weight_kg'].isna().sum())
missing_declared_value_before = int(raw_dedup['declared_value'].isna().sum())
missing_payment_mode_before = int(raw_dedup['payment_mode'].isna().sum())
sentinel_weight_rows_before = int((raw_dedup['parcel_weight_kg']>100).sum())
service_type_spellings_before = raw_dedup['service_type'].nunique()

hub_rows_after_merge = len(features)
service_type_spellings_after = int(cleaned["service_type"].nunique())

# Created CSV
features.to_csv("data/modelling_table.csv", index=False)


delivery_hours = features["delivery_hours"]

q1 = float(delivery_hours.quantile(0.25))
q3 = float(delivery_hours.quantile(0.75))
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

hours_outlier_count = int(
    (
        (delivery_hours < lower_bound)
        | (delivery_hours > upper_bound)
    ).sum()
)

nulls_in_features_and_target = int(
    features[FEATURES + ["delivery_hours", "is_delayed"]]
    .isna()
    .sum()
    .sum()
)

delayed_count = int(features["is_delayed"].sum())
delay_rate = float(features["is_delayed"].mean())

declared_value_mean = float(
    features["declared_value_inr"].mean()
)

declared_value_median = float(
    features["declared_value_inr"].median()
)

log_declared_value_mean = float(
    features["log_declared_value"].mean()
)

log_declared_value_median = float(
    features["log_declared_value"].median())

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", raw_dedup_rows),
    ("missing_distance_km_before", missing_distance_before),
    ("missing_parcel_weight_before", missing_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows_before),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", hub_rows_after_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower_bound),
    ("hours_upper_bound", upper_bound),
    ("hours_outlier_count", hours_outlier_count),
    ("declared_value_mean_inr", declared_value_mean),
    ("declared_value_median_inr", declared_value_median),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median),
]
checks_df = pd.DataFrame(checks, columns=["metric", "value"])

checks_df['value']=pd.to_numeric(checks_df['value'], errors='coerce').round(4)
checks_df.to_csv(
    "outputs/part_a_checks.csv",
    index=False,
)


# Delay band
features["distance_band"] = pd.cut(
    features["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

group_results = []

group_definitions = [
    ("service_type", "service_type"),
    ("hub_tier", "hub_tier"),
    ("distance_band", "distance_band"),
]

for dimension, column in group_definitions:
    grouped = features.groupby(
        column,
        observed=True,
        sort=True,
    )

    for group, group_df in grouped:
        group_results.append({
            "dimension": dimension,
            "group": group,
            "shipments": len(group_df),
            "delayed": int(group_df["is_delayed"].sum()),
            "delay_rate": group_df["is_delayed"].mean(),
            "mean_hours": group_df["delivery_hours"].mean(),
        })

delay_by_group = pd.DataFrame(
    group_results,
    columns=[
        "dimension",
        "group",
        "shipments",
        "delayed",
        "delay_rate",
        "mean_hours",
    ],
)

delay_by_group["delay_rate"] = (
    delay_by_group["delay_rate"].round(4)
)

delay_by_group["mean_hours"] = (
    delay_by_group["mean_hours"].round(4)
)

delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False,
)


numeric_columns = features.select_dtypes(
    include=[np.number]
)

correlations = numeric_columns.corr().round(4)

correlations.to_csv(
    "outputs/part_a_correlations.csv",
    index_label="column",
)

print("Part A completed successfully.")
print(f"Raw rows: {raw_rows}")
print(f"Rows after deduplication: {raw_dedup_rows}")
print(f"Rows after hub merge: {hub_rows_after_merge}")
print(f"Delayed shipments: {delayed_count}")
print(f"Delay rate: {delay_rate:.4f}")
print(f"Missing required values: {nulls_in_features_and_target}")
print(f"Median values: {medians}")
print("Generated:")
print("  data/modelling_table.csv")
print("  outputs/part_a_checks.csv")
print("  outputs/part_a_delay_by_group.csv")
print("  outputs/part_a_correlations.csv")

