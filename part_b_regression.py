import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score


# Load modelling data
df = pd.read_csv("data/modelling_table.csv")

# Separate features and target
X = df[FEATURES]
y = df["delivery_hours"]

# Validate data
assert X.shape[1] == 17, "Expected 17 features"
assert list(X.columns) == FEATURES, "Feature order mismatch"
assert X.isnull().sum().sum() == 0, "Missing values found in features"
assert y.isnull().sum() == 0, "Missing values found in target"

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Second split: divide temporary data equally
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

# Fit scaler only on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test using training statistics
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Calculate maximum absolute column mean
train_scaled_max_abs_mean = abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = abs(
    X_test_scaled.mean(axis=0)
).max()

print("\nScaling checks")
print("Training max absolute mean:", train_scaled_max_abs_mean)
print("Testing max absolute mean:", test_scaled_max_abs_mean)

assert train_scaled_max_abs_mean < 1e-9
assert test_scaled_max_abs_mean > 0.01


# Save validation checks
checks = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {
        "metric": "train_scaled_max_abs_mean",
        "value": round(train_scaled_max_abs_mean, 4)
    },
    {
        "metric": "test_scaled_max_abs_mean",
        "value": round(test_scaled_max_abs_mean, 4)
    }
])

checks.to_csv("outputs/part_b_checks.csv", index=False)

print("\nPart B checks saved.")

# Task 3: Mean baseline

baseline_prediction = y_train.mean()
baseline_predictions = [baseline_prediction] * len(y_val)

baseline_mae = mean_absolute_error(y_val, baseline_predictions)
baseline_mse = mean_squared_error(y_val, baseline_predictions)
baseline_rmse = baseline_mse ** 0.5
baseline_r2 = r2_score(y_val, baseline_predictions)

print("\nMean Baseline (Validation)")
print("Prediction:", baseline_prediction)
print("MAE:", baseline_mae)
print("MSE:", baseline_mse)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)

# Keep this row for part_b_metrics.csv in Task 5
metrics_rows = [{
    "model": "mean_baseline",
    "split": "val",
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "RMSE": baseline_rmse,
    "R2": baseline_r2
}]

# Task 4: Decision tree depth diagnostic

depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_depth_results = []

for depth in depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_predictions = tree.predict(X_train)
    val_predictions = tree.predict(X_val)

    train_rmse = mean_squared_error(
        y_train, train_predictions
    ) ** 0.5

    val_rmse = mean_squared_error(
        y_val, val_predictions
    ) ** 0.5

    tree_depth_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

# Save depth diagnostic
tree_depth_df = pd.DataFrame(tree_depth_results)

tree_depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False,
    float_format="%.4f"
)

print("\nDecision Tree Depth Diagnostic")
print(tree_depth_df.to_string(index=False))

# Select depth with lowest validation RMSE.
# If tied, select the smaller depth.
best_tree_result = min(
    tree_depth_results,
    key=lambda row: (
        row["val_rmse"],
        float("inf") if row["max_depth"] == "unlimited"
        else row["max_depth"]
    )
)

best_tree_depth = best_tree_result["max_depth"]

print("\nBest tree depth:", best_tree_depth)
print("Best validation RMSE:", best_tree_result["val_rmse"])

# Task 5: Train and compare four regression models

# Convert selected depth into an integer or None
selected_depth = (
    None if best_tree_depth == "unlimited"
    else int(best_tree_depth)
)

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=selected_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

# Store fitted models for Task 7
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_val, predictions)

    metrics_rows.append({
        "model": name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# Save validation metrics
metrics_df = pd.DataFrame(metrics_rows)

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
    float_format="%.4f"
)

print("\nRegression Model Comparison")
print(metrics_df.to_string(index=False))

# Rank candidate models by validation RMSE
candidate_metrics = metrics_df[
    metrics_df["model"].isin(models.keys())
].sort_values("RMSE")

print("\nModel Ranking (Validation RMSE)")
print(candidate_metrics[["model", "RMSE", "R2"]].to_string(index=False))

# Task 6: Gradient Boosting learning-rate experiment

learning_rates = [0.01, 0.1, 1.0]

gb_results = []

for lr in learning_rates:

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    val_rmse = mean_squared_error(
        y_val, predictions
    ) ** 0.5

    val_r2 = r2_score(y_val, predictions)

    gb_results.append({
        "learning_rate": lr,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

# Save learning-rate results
gb_df = pd.DataFrame(gb_results)

gb_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
    float_format="%.4f"
)

print("\nGradient Boosting Learning-Rate Comparison")
print(gb_df.to_string(index=False))

# Task 7: Cross-validation

# Combine training and validation data (1,020 rows)
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Select two best models using validation RMSE
top_two = candidate_metrics.head(2)["model"].tolist()

# One shared KFold object for both models
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name in top_two:

    scores = cross_val_score(
        models[name],
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False,
    float_format="%.4f"
)

print("\nCross-validation Results")
print(cv_df.to_string(index=False))

# Select final model using validation RMSE
winner = candidate_metrics.iloc[0]["model"]

final_model = models[winner]

print("\nFinal selected model:", winner)

# Final evaluation on the test set (one time only)
test_predictions = final_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, test_predictions)

test_row = {
    "model": winner,
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}

print("\nFinal Test Results")
print(test_row)

# Add final test result to validation metrics
metrics_rows.append(test_row)

final_metrics_df = pd.DataFrame(metrics_rows)

final_metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
    float_format="%.4f"
)

print("\nFinal Part B metrics saved.")
print(final_metrics_df.to_string(index=False))