# Freight Rate Prediction Challenge

See `freight-rate-ml-assessment.pdf` for the original assessment instructions.

## Repository layout

```
data/train-test.csv                       labeled development data (Jan-Oct 2025)
data/validation.csv                       12,000 loads needing predictions (load_id + features)
data/validation-predictions-template.csv  template to fill (load_id, predicted_rate)
data/december-chart-inputs.csv            31 rows, one per December day, fixed route
score.py                                  provided scorer/validator (unchanged)

src/features.py                      shared data cleaning + feature engineering
src/model.py                         model definition (HistGradientBoostingRegressor)
train.py                             trains, validates, and writes both prediction files
eda.py                               regenerates the figures used in the report
generate_report.py                   builds reports/Freight_Rate_Report.pdf

reports/                             holdout metrics, trained model, figures, PDF report
scorer_results/                      output of score.py (candidate_december.png)
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
# 1. Clean data, validate on a time-based holdout, fit the final model,
#    and write validation_predictions.csv + december_predictions.csv
python train.py

# 2. Validate both output files and render the December chart
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv

# 3. (optional) regenerate the EDA/diagnostic figures and the PDF report
python -m pip install -r requirements-report.txt
python eda.py
python generate_report.py
```

`train.py` prints holdout MAE/RMSE/MAPE/R2 and saves them to
`reports/holdout_metrics.json`. The full write-up of the approach, data
quality issues found, model selection and the December chart is in
`reports/Freight_Rate_Report.pdf`.

## Approach summary

- **Validation split**: time-based (train on Jan-Aug 2025, hold out Sep-Oct
  2025), because the task is forecasting dates never seen in training
  (data/validation.csv is Nov-Dec 2025; the December chart is Dec 2025). A
  random K-fold split would overstate accuracy.
- **Features**: distance, weight (sign-flip errors fixed via `abs()`),
  pickup/delivery lat/lon, equipment, and cyclical (sin/cos) date encodings.
  `market_index` and `quote_signal` are excluded — they're missing from
  `data/december-chart-inputs.csv` and showed negligible predictive value in
  testing, so dropping them keeps one consistent pipeline for both outputs.
  City names are intentionally *not* used as a categorical feature since 8
  of the 72 cities in `data/validation.csv` never appear in training; lat/lon
  generalizes to unseen cities instead.
- **Model**: `HistGradientBoostingRegressor` (absolute-error loss), chosen
  after comparing against Linear Regression, Random Forest, Gradient
  Boosting, and XGBoost on the same holdout.

See `reports/Freight_Rate_Report.pdf` for full details, figures, and the
December prediction chart.
