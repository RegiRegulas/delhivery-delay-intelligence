# Delhivery Last-Mile Delay Intelligence Pipeline

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the completed scripts from the project root:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
```

## Data findings

**(i) Delay by service type:** Express had the highest delay rate (0.4897), followed by Standard (0.2056) and Economy (0.0857). Express promises delivery within 36 hours, and its mean delivery time was 36.8056 hours, slightly above its promise. Economy promises delivery within 96 hours, while its mean delivery time was 82.0931 hours, comfortably below its promise. Delays are measured against each service type's own promised delivery time.

**(ii) Delay by hub tier:** Delay rates increased across tiers: Tier 1 (0.0400), Tier 2 (0.2150), and Tier 3 (0.4400).

**(iii) Delay by distance band:** Delay rates increased with distance: 0–200 km (0.0219), 201–400 km (0.1368), 401–600 km (0.4364), and 601+ km (0.8523).

**(iv) Correlation with delivery hours:** The strongest positive correlations were promised hours (0.8372), distance (0.4013), and number of stops (0.3247). Hub daily capacity had a negative correlation (-0.3237).

**(v) Log transformation of declared value:** Before transformation, `declared_value_inr` had a mean of INR 1,533.05 and a median of INR 1,121.50. The higher mean suggests a right-skewed distribution, with some high-value shipments pulling the mean upward. After log transformation, the mean (7.0277) and median (7.0224) were much closer, indicating reduced skew and a more balanced distribution.

The overall delay rate was 0.2317, with 278 delayed shipments out of 1,200.

Generated outputs:
- `data/modelling_table.csv`
- `outputs/part_a_checks.csv`
- `outputs/part_a_delay_by_group.csv`
- `outputs/part_a_correlations.csv`

## Model results (regression)

Five approaches were evaluated: a mean baseline, Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.

- **Baseline:** The validation R² was -0.0212. A constant prediction using the training mean explains almost none of the variation in delivery hours. The slightly negative value occurs because the training mean differs from the validation mean.
- **Best validation model:** Linear Regression achieved an RMSE of 4.8912 hours and an R² of 0.9313.
- **Tree depth:** Depth 1 had high bias (train RMSE 11.6334, validation RMSE 12.1324), with both errors high and a small gap. The unlimited tree had high variance (train RMSE 0.0000, validation RMSE 7.7890), showing a large generalization gap. Depth 6 gave the lowest validation RMSE of 6.2796 hours.
- **Learning rate:** A learning rate of 0.1 performed best (R² = 0.9306). At 0.01, R² was 0.7127 because learning was too slow and the model underfit. At 1.0, R² was 0.8440 because the large updates reduced generalization performance.
- **Cross-validation:** Gradient Boosting achieved a mean R² of 0.9352 (std 0.0044), compared with Linear Regression's 0.9261 (std 0.0048).

The validation RMSE difference between Linear Regression (4.8912) and Gradient Boosting (4.9165) was small. Although Linear Regression wins under the brief's validation-RMSE selection rule, Gradient Boosting performs better across the five cross-validation folds. I would favour Gradient Boosting for deployment because its cross-validation result provides stronger evidence of generalization across different data splits. It is more complex and less interpretable than Linear Regression, so this choice trades simplicity for the ability to capture nonlinear relationships and feature interactions.

The capstone's validation-based final model was Linear Regression. It achieved a test RMSE of 4.5235 hours and an R² of 0.9493. Gradient Boosting was not evaluated on the test set, so its test performance cannot be compared directly.

## Model results (classification)

## Operations recommendation

## Production plan

## Limitations

The analysis uses synthetic data and may not reflect real-world delivery operations. Correlation does not establish causation.