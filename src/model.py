"""Model definition shared between training/evaluation and the final fit."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_FEATURES

# Chosen via a time-based holdout comparison against LinearRegression,
# RandomForest, GradientBoosting and XGBoost (see reports/Freight_Rate_Report.pdf).
# HistGradientBoostingRegressor with an absolute-error objective gave the
# best MAE/MAPE and was competitive on RMSE, and is robust to the small
# fraction of extreme-value "spot surge" loads in the training data.
MODEL_PARAMS = dict(
    loss="absolute_error",
    max_iter=400,
    learning_rate=0.06,
    max_depth=6,
    random_state=42,
)


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [("equipment", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    return Pipeline([("preprocess", preprocess), ("model", model)])
