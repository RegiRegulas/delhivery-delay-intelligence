# Delhivery Last-Mile Delay Intelligence Pipeline

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts completed so far:

```bash
python generate_data.py
python part_a_clean_eda.py
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

## Model results (classification)

## Operations recommendation

## Production plan

## Limitations

The analysis uses synthetic data and may not reflect real-world delivery operations. Correlation does not establish causation.