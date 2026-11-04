
import os

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Setup
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

modelling_df = pd.read_csv("data/modelling_table.csv")
hubs_df = pd.read_csv("data/hubs.csv")


# --------------------------------------------------
# Task 1: Build hub profile
# --------------------------------------------------

hub_profile = (
    modelling_df
    .groupby("hub_id")
    .agg(
        shipments=("is_delayed", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)

expected_profile_columns = [
    "hub_id",
    "shipments",
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

hub_profile = hub_profile[expected_profile_columns]

assert len(hub_profile) == 24
assert (hub_profile["shipments"] == 50).all()

hub_profile.to_csv(
    "outputs/hub_profile.csv",
    index=False,
    float_format="%.4f"
)

print("Hub Profile")
print(hub_profile.to_string(index=False))


# --------------------------------------------------
# Task 2: K-Means segmentation
# --------------------------------------------------

profile_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

# Hubs must remain in ascending hub_id order.
X_hubs = hub_profile[profile_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hubs)

# Business requirement: exactly three segments.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

hub_profile["cluster"] = cluster_labels


# --------------------------------------------------
# Task 3: Separation evidence
# --------------------------------------------------

distance_matrix = pairwise_distances(
    X_scaled,
    metric="euclidean"
)

# Only unordered pairs: upper triangle, excluding diagonal.
upper_i, upper_j = np.triu_indices(
    len(hub_profile),
    k=1
)

pair_distances = distance_matrix[upper_i, upper_j]

separation_results = []

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    same_cluster = (
        labels[upper_i] == labels[upper_j]
    )

    different_cluster = ~same_cluster

    within_mean = pair_distances[same_cluster].mean()
    across_mean = pair_distances[different_cluster].mean()

    across_over_within = across_mean / within_mean

    cluster_sizes = sorted(
        np.bincount(labels).tolist(),
        reverse=True
    )

    separation_results.append({
        "K": k,
        "across_over_within": across_over_within,
        "cluster_sizes": "-".join(
            str(size) for size in cluster_sizes
        )
    })

separation_df = pd.DataFrame(separation_results)

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False,
    float_format="%.4f"
)

print("\nK-Means Separation Results")
print(separation_df.to_string(index=False))


# --------------------------------------------------
# Task 4: Name segments by delay rate
# --------------------------------------------------

cluster_delay_rates = (
    hub_profile
    .groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

# Lowest delay rate -> Standard
# Middle delay rate -> Watch
# Highest delay rate -> Escalate
segment_names = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate"
}

hub_profile["segment"] = (
    hub_profile["cluster"].map(segment_names)
)

# Attach city and tier metadata.
hub_segments = (
    hub_profile[["hub_id", "segment"]]
    .merge(
        hubs_df[["hub_id", "hub_city", "hub_tier"]],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

assert len(hub_segments) == 24
assert hub_segments[
    ["segment", "hub_city", "hub_tier"]
].notna().all().all()

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False,
    float_format="%.4f"
)

# Segment profile: means of hub-level values.
segment_profile = (
    hub_profile
    .groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

# Keep the requested business order.
segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = (
    segment_profile
    .sort_values("segment")
    .reset_index(drop=True)
)

segment_profile["segment"] = (
    segment_profile["segment"].astype(str)
)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False,
    float_format="%.4f"
)

# Tier-to-segment expectation.
tier_segment_mapping = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(tier_segment_mapping)
)

tier_mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
][["hub_id", "segment", "hub_city", "hub_tier"]].copy()

tier_mismatch.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# Summary and acceptance checks
# --------------------------------------------------

print("\nHub Segments")
print(hub_segments.to_string(index=False))

print("\nSegment Profile")
print(segment_profile.to_string(index=False))

print("\nTier vs Segment Mismatches")
print(tier_mismatch.to_string(index=False))

print("\nPart D segmentation completed.")
