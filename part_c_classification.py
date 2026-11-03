import pandas as pd
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES, NOISE_COLUMNS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_classifier(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        )
    }

# Load modelling data
df = pd.read_csv("data/modelling_table.csv")

# Features and classification target
X = df[FEATURES]
y = df["is_delayed"]

# Validate data
assert list(X.columns) == FEATURES
assert X.shape[1] == 17
assert X.isnull().sum().sum() == 0
assert y.isnull().sum() == 0

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Second split: 50% validation, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

print("\nDelayed parcels:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Testing:", int(y_test.sum()))

assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (195, 42, 41)

# Task 2A: Blind baseline
baseline_predictions = [0] * len(y_val)

baseline_result = evaluate_classifier(
    y_val,
    baseline_predictions
)

models_results = [{
    "model": "blind_all_on_time",
    **baseline_result
}]

# Task 2B: Logistic Regression

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_result = evaluate_classifier(
    y_val,
    logistic_predictions
)

models_results.append({
    "model": "logistic_regression",
    **logistic_result
})

# Task 2C: Decision Tree depth selection

tree_depths = [2, 3, 4, 5, 6, 8, 10]

tree_results = []

for depth in tree_depths:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    predictions = tree.predict(X_val)

    result = evaluate_classifier(y_val, predictions)

    tree_results.append({
        "depth": depth,
        "f1": result["f1"],
        "model": tree,
        "metrics": result
    })

best_tree = max(
    tree_results,
    key=lambda result: (result["f1"], -result["depth"])
)

best_tree_depth = best_tree["depth"]

models_results.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **best_tree["metrics"]
})

print("\nDecision Tree Depth Results")

for result in tree_results:
    print(
        f"Depth {result['depth']}: "
        f"F1 = {result['f1']:.4f}"
    )

print("Selected depth:", best_tree_depth)

# Task 2D: Default Random Forest

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_val)

rf_result = evaluate_classifier(
    y_val,
    rf_predictions
)

models_results.append({
    "model": "random_forest_default",
    **rf_result
})

# Save Task 2 results

models_df = pd.DataFrame(models_results)

models_df.to_csv(
    "outputs/part_c_models.csv",
    index=False,
    float_format="%.4f"
)

print("\nClassification Model Comparison")
print(models_df.to_string(index=False))

# Task 3A: Feature importance using Random Forest

importance_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

importance_model.fit(X_train, y_train)

rf_importances = importance_model.feature_importances_

# Task 3B: ANOVA F-scores
selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(X_train, y_train)

f_scores = selector_all.scores_

# Build feature score table
feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances
})

# Rank features by F-score (1 = highest)
feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# Mark noise columns
feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

# Arrange columns as required
feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

feature_scores = feature_scores.sort_values("f_rank")

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
    float_format="%.4f"
)

print("\nFeature Importance and F-score Ranking")
print(feature_scores.to_string(index=False))

# Calculate combined noise importance
noise_importance = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()

print("\nCombined noise feature importance:", noise_importance)

# Task 3C: SelectKBest comparison

k_values = [5, 7, 10, 12, 17]

selectk_results = []

for k in k_values:

    model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    val_f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    selectk_results.append({
        "k": k,
        "val_f1": val_f1
    })

selectk_df = pd.DataFrame(selectk_results)

selectk_df.to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
    float_format="%.4f"
)

print("\nSelectKBest Comparison")
print(selectk_df.to_string(index=False))

# Task 4: Feature selection using k=7

selector = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

print("\nSelected features:")
print(X.columns[selector.get_support()].tolist())

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(X_train_selected, y_train)

grid_predictions = grid_search.best_estimator_.predict(
    X_val_selected
)

grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0
)

print("\nGrid Search")
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)
print("Validation F1:", grid_val_f1)

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_selected, y_train)

random_predictions = random_search.best_estimator_.predict(
    X_val_selected
)

random_val_f1 = f1_score(
    y_val,
    random_predictions,
    zero_division=0
)

print("\nRandomized Search")
print("Best parameters:", random_search.best_params_)
print("Best CV accuracy:", random_search.best_score_)
print("Validation F1:", random_val_f1)

default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest.fit(X_train_selected, y_train)

default_predictions = default_forest.predict(
    X_val_selected
)

default_val_f1 = f1_score(
    y_val,
    default_predictions,
    zero_division=0
)

print("\nDefault Forest (7 features)")
print("Validation F1:", default_val_f1)

tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": None,
        "val_f1": default_val_f1
    }
]

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False,
    float_format="%.4f"
)

print("\nHyperparameter Tuning Results")
print(tuning_df.to_string(index=False))

# Task 5: Select the final model using validation F1.
# Tie-breaking priority:
# 1. default_forest_same_7_features
# 2. grid_search
# 3. randomized_search

candidate_models = {
    "default_forest_same_7_features": {
        "model": default_forest,
        "val_f1": f1_score(
            y_val,
            default_forest.predict(X_val_selected),
            zero_division=0
        ),
        "priority": 0
    },
    "grid_search": {
        "model": grid_search.best_estimator_,
        "val_f1": f1_score(
            y_val,
            grid_search.best_estimator_.predict(X_val_selected),
            zero_division=0
        ),
        "priority": 1
    },
    "randomized_search": {
        "model": random_search.best_estimator_,
        "val_f1": f1_score(
            y_val,
            random_search.best_estimator_.predict(X_val_selected),
            zero_division=0
        ),
        "priority": 2
    }
}

chosen_from = sorted(
    candidate_models,
    key=lambda name: (
        -candidate_models[name]["val_f1"],
        candidate_models[name]["priority"]
    )
)[0]

chosen_params = (
    candidate_models[chosen_from]["model"].get_params()
)

print("\nFinal Model Selection")
print("Selected model:", chosen_from)
print("Validation F1:", candidate_models[chosen_from]["val_f1"])

# Build the final pipeline using the original 17 features.
final_model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(
        n_estimators=chosen_params["n_estimators"],
        max_depth=chosen_params["max_depth"],
        min_samples_leaf=chosen_params["min_samples_leaf"],
        random_state=42
    ))
])

# Fit only on training data.
final_model.fit(X_train, y_train)

# Validation probabilities for threshold selection.
val_probabilities = final_model.predict_proba(X_val)[:, 1]

# Task 6: Evaluate thresholds from 0.1 to 0.9.

threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

threshold_results = []

for threshold in threshold_values:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    metrics = evaluate_classifier(y_val, val_predictions)

    threshold_results.append({
        "threshold": threshold,
        **metrics
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False,
    float_format="%.4f"
)

print("\nThreshold Results")
print(threshold_df.to_string(index=False))

# Operations rule:
# Select the highest threshold with validation recall >= 0.80.
eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]

if not eligible_thresholds.empty:
    chosen_threshold = eligible_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1

print("\nChosen Alert Threshold:", chosen_threshold)

# Task 7: Final test evaluation.
# Do not use the test set to tune the model or threshold.

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = evaluate_classifier(y_test, test_predictions)

final_test_result = pd.DataFrame([{
    "threshold": chosen_threshold,
    **test_metrics
}])

final_test_result.to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
    float_format="%.4f"
)

print("\nFinal Test Results")
print(final_test_result.to_string(index=False))


# Task 8: Save the final model configuration.

final_model_params = final_model.named_steps["model"].get_params()

final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": final_model_params["n_estimators"],
    "max_depth": (
        final_model_params["max_depth"]
        if final_model_params["max_depth"] is not None
        else "unlimited"
    ),
    "min_samples_leaf": final_model_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)

print("\nFinal Configuration")
print(final_config.to_string(index=False))

print("\nPart C completed.")
