# Loom outline (2-3 min)

Record this yourself — a script for a video narrated in your own voice is
here so you don't have to improvise on camera; the assessment asks for a
walkthrough from you, not a written recording.

**1. Key findings from exploring the data (~30s)**
- 48,000 loads, Jan-Oct 2025, 64 cities, 3 equipment types.
- distance correlates 0.91 with posted_rate — by far the strongest signal.
- Clear monthly seasonality: rate/mile peaks in June (~$2.33/mi), dips
  into ~$2.19/mi by August (show `reports/figures/monthly_seasonality.png`).
- market_index & quote_signal barely correlate with rate (<0.07) — turned
  out to be low-value features once tested.

**2. Data-quality issues found and how addressed (~30s)**
- 292 rows had negative `weight` — a sign-flip entry error, identical
  distribution to positive weights once flipped back. Fixed with `abs()`.
- ~300 missing weights (train), 165 (validation) — filled with the
  training-set median.
- 8 of 72 validation cities never appear in training (e.g. Chicago, San
  Diego) — handled by using lat/lon instead of city identity as a feature.
- A small tail of loads (~0.7%) have 2-7x the normal rate/mile with no
  explanatory feature — likely real spot-market spikes, kept in training,
  called out as inherent noise.

**3. Reasoning behind the chosen model (~30s)**
- Compared Linear Regression, Random Forest, Gradient Boosting, XGBoost,
  and HistGradientBoostingRegressor on the same holdout.
- HistGradientBoostingRegressor (absolute-error loss) won on MAE/MAPE and
  handles the outlier tail better than a squared-error objective.

**4. Training/validation approach (~30s)**
- Time-based split: train on Jan-Aug, hold out Sep-Oct, because the real
  task is forecasting Nov/Dec — dates the model has never seen. A random
  split would've overstated accuracy.
- Holdout: MAE $118.82, MAPE 5.2%, R^2 0.826 (show
  `reports/figures/actual_vs_predicted.png`).
- Also stress-tested by fully removing 8 cities from training to confirm
  the lat/lon approach generalizes to unseen cities (~15% MAE increase,
  not a collapse).

**5. Code walkthrough (~30-45s)**
- `src/features.py`: one shared cleaning/feature-engineering path used
  identically for train, validation and December — no train/serve skew.
- `src/model.py`: the model + hyperparameters.
- `train.py`: holdout evaluation -> refit on all data -> writes both
  prediction CSVs.
- `score.py` output: show the December chart
  (`scorer_results/candidate_december.png`) and mention the narrow
  ~$772-790 range reflects low date-feature importance, not a strong
  seasonal signal.
