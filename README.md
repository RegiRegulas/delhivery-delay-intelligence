# Delhivery Last-Mile Delay Intelligence Pipeline

## How to run

### 1. Set up the environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

Run the scripts from the project root in this order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

### 3. Model packaging

`python package_model.py` rebuilds the final Part C classification pipeline using the saved configuration and original stratified training split. It regenerates:

- `models/delay_model.pkl`
- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`
- `outputs/part_d_reload_check.csv`

The committed `delay_model.pkl` loads reliably with the pinned scikit-learn version in `requirements.txt`. Pickle files are not guaranteed to be compatible across scikit-learn versions. If the installed version differs, install the pinned dependencies and rerun `python package_model.py` to regenerate the model.

### 4. Prediction output

`python predict_delay.py` loads the packaged model and preprocessing metadata, scores the 48 new shipments, and writes:

```text
outputs/new_shipment_actions.csv
```

The output includes shipment ID, hub ID, hub segment, delay probability, flag, and recommended operational action.

---

## Data findings

The cleaned modelling table contains 1,200 shipments and 17 model features.

Data-quality checks:

- Raw records: 1,224.
- Duplicate records removed: 24.
- Records after deduplication: 1,200.
- Missing values before cleaning: distance 36, parcel weight 24, declared value 12, and payment mode 18.
- Invalid sentinel weight rows (`sentinel_weight_rows`): 5.
- Service-type spelling inconsistencies before normalization: 12; after normalization: 3.
- Historical imputation medians: distance 314.95 km, parcel weight 1.83 kg, and declared value INR 1,121.50.
- No missing values remained in the modelling features or target.

Overall target distribution:

- Delayed shipments: 278.
- On-time shipments: 922.
- Overall delay rate: 0.2317.

### (i) Most-delayed service type

| Service | Shipments | Delayed | Delay rate | Mean delivery hours |
|---|---:|---:|---:|---:|
| Express | 243 | 119 | 0.4897 | 36.8056 |
| Standard | 642 | 132 | 0.2056 | 52.2569 |
| Economy | 315 | 27 | 0.0857 | 82.0931 |

Express is the most-delayed service type, with a delay rate of 0.4897. Its mean delivery time of 36.8056 hours is above its 36-hour promise. Economy has a lower delay rate of 0.0857, and its mean delivery time of 82.0931 hours is below its 96-hour promise.

### (ii) Hub-tier trend

| Hub tier | Delay rate |
|---|---:|
| Tier 1 | 0.0400 |
| Tier 2 | 0.2150 |
| Tier 3 | 0.4400 |

The delay rate increases with hub tier. Tier 3 hubs have the highest observed delay rate, while Tier 1 hubs have the lowest.

### (iii) Distance-band trend

| Distance band (km) | Shipments | Delay rate |
|---|---:|---:|
| 0–200 | 274 | 0.0219 |
| 201–400 | 563 | 0.1368 |
| 401–600 | 275 | 0.4364 |
| 601+ | 88 | 0.8523 |

The observed delay rate increases substantially with distance. Shipments in the 601+ km band have the highest delay rate at 0.8523.

### (iv) Top three correlations

The three strongest correlations with delivery hours, ordered by absolute magnitude, are:

| Feature | Correlation |
|---|---:|
| promised_hours | 0.8372 |
| distance_km | 0.4013 |
| n_stops | 0.3247 |

These are associations in the synthetic dataset and do not establish independent causal effects.

### (v) Log transformation

Declared value had a mean of INR 1,533.05 and a median of INR 1,121.50. After the logarithmic transformation, the mean and median were 7.0277 and 7.0224. The smaller gap indicates that the transformation reduced the influence of the right-skewed distribution.

Delivery-hours distribution checks:

- Q1: 43.2550 hours.
- Q3: 72.5625 hours.
- IQR lower bound: -0.7063 hours.
- IQR upper bound: 116.5238 hours.
- Outlier count: 1.

---

## Model results (regression)

The regression task predicts delivery hours.

### Evaluation setup

- Features: 17 columns from the shared feature list.
- Split: 840 training, 180 validation, and 180 test rows.
- Random state: 42.
- StandardScaler was fitted on training data only.
- Model selection was based on validation RMSE.

### Validation results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Mean baseline | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| Linear Regression | 3.9222 | 23.9235 | 4.8912 | 0.9313 |
| Decision Tree (depth 6) | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| Random Forest | 4.2505 | 29.8987 | 5.4680 | 0.9141 |
| Gradient Boosting | 3.9166 | 24.1722 | 4.9165 | 0.9306 |

The mean baseline predicts the training-set mean for every shipment. Its slightly negative validation R² of -0.0212 occurs because the training mean differs slightly from the validation mean. The constant prediction therefore explains none of the validation target variance.

Linear Regression achieved the lowest validation RMSE of 4.8912, narrowly ahead of Gradient Boosting at 4.9165.

### Decision-tree depth and bias-variance diagnostics

| Tree depth | Training RMSE | Validation RMSE |
|---|---:|---:|
| 1 | 11.6334 | 12.1324 |
| 2 | 9.3687 | 10.4114 |
| 3 | 7.5295 | 8.7033 |
| 4 | 6.0999 | 7.3896 |
| 6 | 4.7114 | 6.2796 |
| 8 | 3.3360 | 6.8385 |
| 12 | 1.3613 | 7.4173 |
| Unlimited | 0.0000 | 7.7890 |

Depth 1 shows high bias: both training and validation RMSE are high, and the small train-validation gap indicates that the model underfits both datasets.

At unlimited depth, training RMSE reaches 0.0000 while validation RMSE is 7.7890. This large gap indicates high variance and overfitting. Depth 6 achieved the lowest validation RMSE among the tested tree depths.

### Gradient Boosting learning-rate diagnostics

| Learning rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| 0.1 | 4.9165 | 0.9306 |
| 1.0 | 7.3701 | 0.8440 |

A learning rate of 0.1 performed best among the tested values.

At 0.01, the updates are too small, so the model underfits after 100 trees. At 1.0, the updates are too large and overshoot, reducing validation performance.

### Cross-validation and final model

Cross-validation on the 1,020 training-plus-validation rows produced:

| Model | Mean CV R² | CV R² standard deviation |
|---|---:|---:|
| Gradient Boosting | 0.9352 | 0.0044 |
| Linear Regression | 0.9261 | 0.0048 |

The capstone's validation-RMSE selection rule chooses Linear Regression. Its test results were:

- MAE: 3.5671 hours.
- MSE: 20.4624.
- RMSE: 4.5235 hours.
- R²: 0.9493.

Gradient Boosting is the deployment candidate based on its stronger cross-validation mean R² and low variation. Its test performance was not measured in this experiment. The reported regression test metrics belong to Linear Regression, not Gradient Boosting.

---

## Model results (classification)

The classification task predicts whether a shipment will miss its delivery promise.

### Evaluation setup

- Target: `is_delayed`.
- Split: stratified 840 training, 180 validation, and 180 test rows.
- Delayed rows: 195 training, 42 validation, and 41 test.
- Random state: 42.

### Initial validation model comparison

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Always on-time baseline | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 0.7561 | 0.7381 | 0.7470 |
| Random Forest (17 features) | 0.8378 | 0.7381 | 0.7848 |

The always-on-time baseline achieves 0.7667 accuracy because most shipments are on time, but it detects no delayed shipments. Accuracy alone therefore hides its failure to identify delays.

### Feature selection

The seven highest-ranked features were:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

All five synthetic noise columns ranked outside the top seven. Their combined Random Forest feature importance was 0.2244.

Validation F1 by selected feature count:

| Selected features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

The seven-feature configuration performed best among the tested values.

### Hyperparameter tuning

Grid Search and Randomized Search were evaluated against the untuned default Random Forest using the selected seven features.

- Grid Search: 27 candidates, 5-fold cross-validation, 135 fits; validation F1 0.7750.
- Randomized Search: 9 candidates, 5-fold cross-validation, 45 fits; validation F1 0.7848.
- Untuned default Random Forest with seven features: validation F1 0.8333.

Neither search beat the untuned forest on validation: 0.7750 and 0.7848 compared with 0.8333. The validation set contains only 42 delayed shipments, so differences of a few correctly classified parcels can change F1 noticeably. These small validation-set differences may largely reflect sampling noise rather than a reliable improvement from tuning.

The default forest with seven features was retained because it achieved the highest validation F1 among the compared configurations.

Final configuration:

- Selector: `SelectKBest(f_classif, k=7)`.
- Classifier: `RandomForestClassifier`.
- Estimators: 100.
- Maximum depth: unlimited.
- Minimum samples per leaf: 1.
- Random state: 42.

### Alert threshold selection

The operational rule was to choose the highest validation threshold that retained recall of at least 0.80.

The selected threshold was **0.50**:

- Validation precision: 0.7955.
- Validation recall: 0.8333.
- Validation F1: 0.8140.

At threshold 0.60, precision increased to 0.8889, but recall fell to 0.7619, below the required 0.80.

### Final test results

At threshold 0.50:

| Metric | Test result |
|---|---:|
| TN | 125 |
| FP | 14 |
| FN | 10 |
| TP | 31 |
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 | 0.7209 |

On the test set, precision of 0.6889 means approximately 7 of every 10 flagged shipments were actually delayed. Recall of 0.7561 means approximately 8 of every 10 delayed shipments were detected.

Validation recall was 0.8333 against test recall of 0.7561. The test set contains only 41 delayed shipments, so each additional missed delayed shipment changes recall by approximately 0.0244.

---

## Operations recommendation

### 1. Escalate hubs

The nine hubs assigned to the Escalate segment have a mean hub-level delay rate of **0.4600**.

| Hub | Delay rate |
|---|---:|
| HUB13 | 0.4600 |
| HUB15 | 0.3600 |
| HUB17 | 0.4600 |
| HUB18 | 0.3800 |
| HUB19 | 0.6200 |
| HUB20 | 0.4800 |
| HUB21 | 0.4000 |
| HUB23 | 0.4400 |
| HUB24 | 0.5400 |

These hubs should receive priority operational review. HUB19 and HUB24 have the highest observed delay rates in this segment.

### 2. Tier-to-segment mismatches

The expected tier mapping was Tier 1 → Standard, Tier 2 → Watch, and Tier 3 → Escalate. Four hubs did not follow this expectation:

| Hub | City | Tier | Actual segment | Interpretation |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | Tier 1 hub behaves like Watch |
| HUB13 | Nagpur | 2 | Escalate | Tier 2 hub behaves like Escalate |
| HUB15 | Kochi | 2 | Escalate | Tier 2 hub behaves like Escalate |
| HUB22 | Dehradun | 3 | Watch | Tier 3 hub behaves like Watch |

A tier label is a starting expectation, not an operational playbook. The observed segment should inform hub-level action because the clustering reflects delivery performance, distance, and stop patterns rather than tier alone.

### 3. Classification threshold and operational meaning

The selected probability threshold is **0.50**.

On the held-out test set:

- Precision 0.6889: approximately 7 out of 10 flagged parcels are delayed.
- Recall 0.7561: approximately 8 out of 10 delayed parcels are caught.

The threshold balances the need to detect delayed parcels against the cost of unnecessary interventions. It should be reviewed as more labelled delivery outcomes become available.

### 4. New shipment actions

The model scored 48 previously unseen shipments:

| Action | Count | Share |
|---|---:|---:|
| Escalate to hub manager | 6 | 0.1250 |
| Proactive customer call | 2 | 0.0417 |
| No action | 40 | 0.8333 |
| **Total** | **48** | **1.0000** |

Eight shipments were flagged, of which six were assigned to hub-manager escalation because they belonged to the Escalate segment. Two flagged shipments in other segments were assigned proactive customer calls.

These are model predictions and recommended actions, not confirmed delivery outcomes.

### 5. Why K=3 was retained

The across-cluster-to-within-cluster distance ratio increased as K increased:

| K | Across / within ratio | Cluster sizes |
|---:|---:|---|
| 2 | 2.8940 | 15-9 |
| 3 | 5.3928 | 9-8-7 |
| 4 | 6.3047 | 8-7-5-4 |
| 5 | 6.8866 | 8-7-4-3-2 |

K=4 and K=5 produce higher separation ratios, but they also create smaller clusters. K=4 contains a cluster of only four hubs, while K=5 contains clusters of three and two hubs. Such small groups are less suitable for defining stable, independent operational playbooks.

The business requirement also specifies three playbooks: Standard, Watch, and Escalate. K=3 provides useful separation while retaining practical group sizes and matching the required operating structure.

---

## Production plan

The current implementation is a prototype built on synthetic data. The following controls describe how it should be operated and improved with real shipment data.

### 1. Map production characteristics to repository components

| Characteristic | Repository implementation | Production consideration |
|---|---|---|
| Accuracy | `outputs/part_c_final_test.csv` and classification metrics | Track precision, recall, and F1 on newly labelled shipments; do not rely on accuracy alone. |
| Reliability | `package_model.py` and `outputs/part_d_reload_check.csv` | Require successful model loading and a reload probability difference no greater than 1e-9. |
| Reproducibility | `requirements.txt`, fixed random states, and deterministic output generation | Pin dependencies and seeds; compare output hashes across repeated runs. |
| Speed | `predict_delay.py` | Measure end-to-end scoring time and monitor it as input volume grows. |
| Maintainability | Separate pipeline scripts and shared `src/shipment_prep.py` | Keep data preparation, modelling, segmentation, packaging, and inference responsibilities separated. |
| Consistency | Shared `clean_shipments()` and `build_features()` functions, plus saved imputation values and feature order | Apply the same historical imputation values, feature definitions, and feature order during training and inference. |
| Monitorability | `outputs/new_shipment_actions.csv`, hub profiles, segment profiles, and evaluation outputs | Record prediction volumes, flagged shares, hub-level delay rates, and performance after actual outcomes are known. |

### 2. Monitoring quantities and proposed triggers

The following are proposed production alert thresholds. They are not results measured by the current synthetic experiment.

| Quantity | Monitoring method | Proposed trigger |
|---|---|---|
| Four-week rolling delayed share | Calculate delayed shipments divided by completed shipments over a rolling four-week window, using matured delivery outcomes | For ordinary non-sale periods, investigate if the rate exceeds 0.2817, which is 0.0500 above the historical 0.2317 rate. Require at least 100 completed shipments in the window. |
| Weekly flagged share | Calculate the proportion of shipments with predicted probability at or above 0.50 | Investigate if the weekly share exceeds 0.30 or falls below 0.05 for two consecutive weeks, with at least 100 scored shipments per week. The current new-shipment reference is 8/48, approximately 0.1667; this small batch is not a stable production baseline. |
| Recall on matured predictions | Compare stored predictions against actual delayed outcomes once delivery labels are available | Review the model if rolling recall falls below 0.70 after at least 30 delayed outcomes have matured. |
| Precision on matured predictions | Compare flagged shipments against confirmed delay labels | Review the threshold or model if rolling precision falls below 0.60 after at least 30 flagged outcomes have matured. |
| Inference runtime | Measure end-to-end execution time in `predict_delay.py` | Investigate if scoring 48 shipments takes more than 5 seconds after the environment is initialized. |

Monitoring windows should include sample-size requirements. A percentage calculated from a handful of shipments is too unstable to justify retraining or changing operational policy.

**Sale-period exception:** The 0.2817 delayed-share trigger applies to ordinary non-sale periods. During the 15–28 September sale window, evaluate delayed share against a sale-specific historical baseline, not the ordinary-period 0.2317 rate. The generator deliberately adds delivery hours for sale parcels and additional hours for Express shipments during the sale, so applying the ordinary-period trigger would produce misleading alerts.

### 3. Retraining schedule

- Monitor predictions and operational outcomes weekly.
- Review model quality monthly once sufficient delivery labels have matured.
- Schedule a full retraining and validation cycle quarterly.
- Trigger an earlier review if the proposed recall, precision, or drift thresholds are breached.
- Use time-based training and validation splits for production retraining so that evaluation better represents future shipments.
- Re-evaluate the alert threshold on a validation period separate from the final test period. Do not tune the threshold on the test set.
- Rebuild and verify the packaged pipeline after retraining, including feature order, imputation values, reload check, and inference output.

A retrained model should not replace the deployed model until it passes the same evaluation and packaging checks.

### 4. September sale-window handling

The 15–28 September sale window should be treated as a distinct operating period because `sale_event_flag` captures sale-event conditions. The data generator also adds delivery hours for sale parcels and an additional increase for Express shipments during the sale.

Compare shipments in this window against previous sale-event periods with similar operating conditions, rather than comparing them directly with ordinary non-sale weeks.

During the sale window:

- Monitor delayed share, flagged share, precision, and recall separately for sale-event shipments.
- Compare shipment distance, number of stops, service-type mix, hub distribution, and hub capacity against comparable sale periods.
- Check whether the relationship between the sale-event indicator and actual delays changes.
- Review whether probability calibration and the 0.50 threshold remain appropriate during the event.
- Retain the sale-event feature during training and inference so that the model can account for event conditions.
- Use a sale-specific delayed-share baseline for alerting; do not apply the ordinary-period 0.2817 trigger to sale weeks.

### 5. Data drift and model drift

Data drift occurs when the distribution of incoming shipment features changes relative to the training data. Relevant signals include:

- A change in the proportion of Express, Standard, or Economy shipments.
- Shifts in distance and number-of-stops distributions.
- Changes in shipment volume across hub tiers or individual hubs.
- Changes in rain and sale-event frequency.
- Changes in the distribution of predicted delay probabilities.
- New or previously unseen hub IDs or changes in hub metadata.

Model drift occurs when predictive performance deteriorates even if input distributions appear stable. It can be identified through falling precision, recall, or F1 once actual delivery outcomes become available.

A change in the sale-event shipment mix should not automatically be treated as model failure. It should first be compared with historical sale-event periods. If the event-conditioned relationship between features and delays has changed, retraining and threshold recalibration should be evaluated.

---

## Limitations

1. **Synthetic data:** The dataset is generated for this capstone and does not establish that the observed relationships or model performance will generalize to real Delhivery operations.

2. **Small test set:** The classification test set contains only 180 shipments and 41 delayed cases. A small number of changed predictions can materially affect precision and recall.

3. **Single-split threshold selection:** The threshold was selected using one validation split. Its performance may vary across time periods, hubs, and changing shipment mixes.

4. **Limited hub segmentation sample:** Hub segments are based on only 24 hubs, with 50 shipments per hub. The resulting clusters and delay rates may change when more shipments or different operating periods are included. The tier-to-segment mismatches should therefore be treated as review signals, not permanent hub classifications.

5. **Unmeasured regression deployment performance:** Gradient Boosting was selected as the deployment candidate based on cross-validation, but its held-out test performance was not measured. The reported regression test metrics belong to Linear Regression, which won the validation-RMSE selection rule.

6. **New shipment outcomes are unknown:** The 48 new shipment predictions have no confirmed delivery labels in this experiment. The flagged count and operational actions cannot be interpreted as measured accuracy or actual delay counts.

7. **Operational thresholds are provisional:** The monitoring triggers and retraining schedule in the production plan are proposed controls. They require validation and adjustment using real operational volumes, costs, and matured delivery outcomes.