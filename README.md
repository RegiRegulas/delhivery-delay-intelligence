# Delhivery Last-Mile Delay Intelligence Pipeline

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline scripts in order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` regenerates the packaged model artifact. Run it after completing the analysis and before using `predict_delay.py`.

## Data findings

The raw dataset contained 1,224 rows. After removing 24 duplicate rows, 1,200 shipments remained.

Missing values were handled using medians calculated from the full historical dataset before the train/validation/test split. These medians describe the input features and do not use information from the prediction target:

- Distance: 314.95 km
- Parcel weight: 1.83 kg
- Declared value: INR 1,121.50

The final modelling table contains 1,200 rows with no missing values across the 17 model features and target. There are 278 delayed shipments, giving an overall delay rate of 0.2317.

**(i) Service type:** Express shipments had a delay rate of 0.4897 (48.97%) and mean delivery time of 36.8056 hours, slightly above their 36-hour promise. Standard shipments had a delay rate of 0.2056 (20.56%). Economy shipments had a delay rate of 0.0857 (8.57%) and mean delivery time of 82.0931 hours, below their 96-hour promise.

**(ii) Hub tier:** Delay rates increased from 0.0400 (4.00%) for Tier 1 hubs to 0.2150 (21.50%) for Tier 2 and 0.4400 (44.00%) for Tier 3.

**(iii) Distance:** Delay rates increased across distance bands:

- 0–200 km: 0.0219 (2.19%)
- 201–400 km: 0.1368 (13.68%)
- 401–600 km: 0.4364 (43.64%)
- 601 km or more: 0.8523 (85.23%)

**(iv) Correlations with delivery hours, ranked by absolute magnitude:**

| Feature | Correlation |
|---|---:|
| promised_hours | 0.8372 |
| distance_km | 0.4013 |
| n_stops | 0.3247 |
| hub_daily_capacity | -0.3237 |
| hub_tier | 0.3175 |

**(v) Declared value transformation:** The original declared-value distribution was right-skewed, with a mean of INR 1,533.0542 and median of INR 1,121.50. After applying the log transformation, the mean was 7.0277 and the median was 7.0224. The much smaller gap between the transformed mean and median indicates that the log transformation pulls in the long right tail and reduces skewness.

Generated outputs include `data/modelling_table.csv`, `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv`, and `outputs/part_a_correlations.csv`.

## Model results (regression)

The regression task predicts delivery hours using 17 features. The dataset was split into 840 training, 180 validation, and 180 test rows. StandardScaler was fitted on training data only.

The mean-prediction baseline achieved a validation RMSE of 18.8552 hours and R² of -0.0212, showing that a constant prediction provides little explanatory power.

### Validation model comparison

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Gradient Boosting | 4.9165 | 0.9306 |
| Random Forest | 5.4680 | 0.9141 |
| Decision Tree (depth 6) | 6.2796 | 0.8867 |

Linear Regression won under the brief's validation-RMSE selection rule. Its test performance was:

- MAE: 3.5671 hours
- MSE: 20.4624
- RMSE: 4.5235 hours
- R²: 0.9493

### Decision tree bias–variance diagnostic

The depth comparison illustrates the bias–variance trade-off:

| Tree depth | Training RMSE | Validation RMSE |
|---|---:|---:|
| 1 | 11.6334 | 12.1324 |
| Unlimited | 0.0000 | 7.7890 |

Depth 1 has a small training/validation gap but high error on both sets, indicating high bias and underfitting. The unlimited tree fits the training data perfectly but has a much larger validation error, indicating high variance and overfitting. Among the tested depths, depth 6 gave the best validation RMSE of 6.2796.

### Gradient Boosting learning rate

| Learning rate | Validation RMSE | Validation R² |
|---:|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of 0.1 performed best. At 0.01, the updates were too small for the fixed number of boosting stages, resulting in underfitting. At 1.0, the larger updates overshot useful improvements and reduced generalization performance.

### Cross-validation and model selection

Five-fold cross-validation on the combined 1,020 training and validation rows produced:

| Model | Mean CV R² | Standard deviation |
|---|---:|---:|
| Gradient Boosting | 0.9352 | 0.0044 |
| Linear Regression | 0.9261 | 0.0048 |

**Candidate to ship: Gradient Boosting.** Although Linear Regression won the required validation-RMSE comparison by a small margin, Gradient Boosting achieved stronger mean cross-validation R² with slightly lower fold-to-fold variation. I would take Gradient Boosting forward as the deployment candidate, subject to a fair held-out test comparison before final production selection. Its test score was not measured in this experiment; the reported test metrics belong to Linear Regression and must not be used as evidence that Gradient Boosting performed better on the test set.

## Model results (classification)

The classification task predicts whether a shipment will miss its delivery promise. The stratified split produced 840 training, 180 validation, and 180 test rows, containing 195, 42, and 41 delayed shipments respectively.

### Initial model comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Blind all-on-time baseline | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The blind baseline achieves 0.7667 accuracy by predicting every shipment as on time. However, it catches 0 of the 42 delayed validation shipments, giving it zero recall. It is useless for the operational purpose of identifying delayed shipments. Accuracy is misleading here because approximately 77% of shipments are on time.

Among the initial trained models, Logistic Regression has the highest recall (0.8333, catching 35 of 42 delayed shipments) and highest F1. The Random Forest has the highest precision (0.8378), but misses 11 of the 42 delayed shipments. Since missing a delayed shipment is more costly than generating an unnecessary alert, recall is an important operational metric.

### Feature importance and noise columns

The combined Random Forest importance of the five noise columns was **0.2244**. Their F-score rankings were all outside the top seven:

- Highest noise F-score: `label_print_batch`, 4.5847.
- Lowest F-score among the top seven: `sale_event_flag`, 36.7281.

The top seven features, in F-score order, were:

1. `distance_km`
2. `hub_tier`
3. `hub_daily_capacity`
4. `n_stops`
5. `promised_hours`
6. `rain_flag`
7. `sale_event_flag`

This indicates that the selected operational features carry substantially stronger univariate signal than the noise columns. Random Forest importance alone can still assign importance to noise features, so the F-score ranking provides a useful complementary check.

### Feature selection

| Number of features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

Selecting seven features produced the best validation F1 of 0.8333. Selecting only five gave the lowest F1 of 0.6988. The seven-feature selector was therefore retained for the final model.

### Hyperparameter tuning

| Search | Candidates | CV accuracy | Validation F1 |
|---|---:|---:|---:|
| Grid Search | 27 | 0.8952 | 0.7750 |
| Randomized Search | 9 | 0.8905 | 0.7848 |
| Default Forest (7 features) | 1 | — | 0.8333 |

Neither tuning method improved validation F1 over the default seven-feature Random Forest. The validation F1 differences are small, and the validation set contains only 180 rows. These differences should not be treated as strong evidence that one configuration will generalize better. The default seven-feature forest was selected because it had the highest validation F1.

The selected configuration was:

- Feature selector: `SelectKBest(f_classif, k=7)`
- Estimator: `RandomForestClassifier`
- Number of trees: 100
- Maximum depth: unlimited
- Minimum samples per leaf: 1
- Random state: 42

### Alert threshold selection

The final model's validation probabilities were evaluated at thresholds from 0.1 to 0.9. The operational rule selected the highest threshold with validation recall of at least 0.80.

The chosen threshold was **0.5**:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.4 | 0.7400 | 0.8810 | 0.8043 |
| 0.5 | 0.7955 | 0.8333 | 0.8140 |
| 0.6 | 0.8889 | 0.7619 | 0.8205 |

At 0.5, the model catches 35 of 42 delayed validation shipments. Raising the threshold to 0.6 improves precision but reduces recall below the required 0.80. The 0.5 threshold therefore satisfies the operational recall constraint while avoiding the larger number of false positives produced by lower thresholds.

### Final test results

The final model and threshold were evaluated once on the held-out test set:

| Metric | Test result |
|---|---:|
| Threshold | 0.5 |
| True negatives | 125 |
| False positives | 14 |
| False negatives | 10 |
| True positives | 31 |
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 | 0.7209 |

The model caught 31 of the 41 delayed test shipments and missed 10. Test recall was 0.7561, compared with validation recall of 0.8333, a decrease of approximately 7.72 percentage points.

This difference should be interpreted cautiously: the test set contains only 41 delayed shipments, so each additional missed shipment changes recall by approximately 2.44 percentage points. The test result is lower than validation performance, but it is the held-out estimate and should not be used to adjust the threshold after evaluation.

## Operations recommendation

Use the seven-feature Random Forest with a 0.5 alert threshold as the classification configuration selected by the specified validation rule.

Prioritize recall when evaluating operational performance because missed delayed shipments are more costly than unnecessary alerts. Monitor false positives as well, since excessive alerts can increase operational workload.

The validation threshold satisfies the minimum 0.80 recall requirement. However, test recall was 0.7561, so the model should be monitored before being relied upon as a production alerting system.

## Production plan

The production plan will be finalized after Part D, when the segment-level findings and operational requirements are available.

The deployment workflow should preserve the fitted preprocessing and model pipeline, validate incoming shipment records, generate delay probabilities, apply the selected alert threshold, and log predictions alongside actual delivery outcomes. Monitoring metrics, numeric alert triggers, retraining cadence, and sale-window handling should be defined from the Part D findings rather than assumed in advance.

## Limitations

- The dataset is synthetic and may not represent real-world shipment patterns or operational complexity.
- Correlation and feature importance do not establish causation.
- The validation and test sets are relatively small, particularly for delayed shipments, so precision and recall estimates can vary.
- The final test recall is lower than validation recall, indicating that validation performance should not be assumed to transfer directly to unseen data.
- The alert threshold of 0.5 was selected using a single validation split. Its performance may vary across different samples or future shipment distributions.
- Gradient Boosting was selected as the regression deployment candidate based on cross-validation, but its held-out test performance was not measured in this experiment.
- The model requires evaluation on real shipment data before production deployment.