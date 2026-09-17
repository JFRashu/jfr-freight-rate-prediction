"""End-to-end pipeline for the Spotter freight rate assessment.

1. Loads and cleans data/train-test.csv
2. Validates the model with a time-based holdout split (train on Jan-Aug
   2025, evaluate on Sep-Oct 2025) since the real task is forecasting
   unseen future dates (Nov validation set, Dec chart).
3. Refits the final model on all available labeled data.
4. Predicts data/validation.csv -> validation_predictions.csv
5. Predicts data/december-chart-inputs.csv -> december_predictions.csv

Run: python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
from src.features import ALL_FEATURES, attach_geo, build_city_lookup, clean_and_engineer, fit_weight_median
from src.model import build_pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"
REPORTS_DIR = ROOT / "reports"
HOLDOUT_CUTOFF = pd.Timestamp("2025-09-01")
TARGET = "posted_rate"


def load_train_test() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def evaluate_holdout(df: pd.DataFrame) -> dict:
    """Time-based split: train on everything before the cutoff, score on
    the most recent slice. This mirrors the real prediction task (Nov/Dec
    dates the model has never seen) far better than a random K-fold split
    would."""
    train_raw = df[df["date"] < HOLDOUT_CUTOFF]
    holdout_raw = df[df["date"] >= HOLDOUT_CUTOFF]

    weight_median = fit_weight_median(train_raw)
    train = clean_and_engineer(train_raw, weight_median)
    holdout = clean_and_engineer(holdout_raw, weight_median)

    pipeline = build_pipeline()
    pipeline.fit(train[ALL_FEATURES], train[TARGET])
    pred = pipeline.predict(holdout[ALL_FEATURES])
    actual = holdout[TARGET].to_numpy()

    metrics = {
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "holdout_start": HOLDOUT_CUTOFF.date().isoformat(),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(mean_squared_error(actual, pred) ** 0.5),
        "mape": float(mean_absolute_percentage_error(actual, pred)),
        "r2": float(r2_score(actual, pred)),
    }
    return metrics


def fit_final_model(df: pd.DataFrame):
    weight_median = fit_weight_median(df)
    full = clean_and_engineer(df, weight_median)
    pipeline = build_pipeline()
    pipeline.fit(full[ALL_FEATURES], full[TARGET])
    return pipeline, weight_median


def predict_validation(pipeline, weight_median: float) -> pd.DataFrame:
    raw = pd.read_csv(VALIDATION_PATH)
    engineered = clean_and_engineer(raw, weight_median)
    predicted_rate = pipeline.predict(engineered[ALL_FEATURES])
    predicted_rate = np.clip(predicted_rate, a_min=1.0, a_max=None)

    template = pd.read_csv(VALIDATION_TEMPLATE_PATH)
    rate_by_id = pd.Series(predicted_rate, index=raw["load_id"].to_numpy())
    template["predicted_rate"] = template["load_id"].map(rate_by_id)
    assert template["predicted_rate"].isna().sum() == 0, "unmatched load_id in validation set"
    return template


def predict_december(pipeline, weight_median: float, city_lookup: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(DECEMBER_PATH)
    raw_geo = attach_geo(raw, city_lookup)
    engineered = clean_and_engineer(raw_geo, weight_median)
    predicted_rate = pipeline.predict(engineered[ALL_FEATURES])
    predicted_rate = np.clip(predicted_rate, a_min=1.0, a_max=None)

    out = raw.copy()
    out["predicted_rate"] = predicted_rate
    return out


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_train_test()

    print("Evaluating on time-based holdout (train < 2025-09-01, holdout >= 2025-09-01)...")
    metrics = evaluate_holdout(df)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    (REPORTS_DIR / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nRefitting final model on all labeled data (Jan-Oct 2025)...")
    pipeline, weight_median = fit_final_model(df)
    joblib.dump({"pipeline": pipeline, "weight_median": weight_median}, REPORTS_DIR / "model.joblib")

    print("Predicting validation.csv -> validation_predictions.csv")
    val_out = predict_validation(pipeline, weight_median)
    val_out.to_csv(ROOT / "validation_predictions.csv", index=False)

    print("Predicting december-chart-inputs.csv -> december_predictions.csv")
    validation_raw = pd.read_csv(VALIDATION_PATH)
    city_lookup = build_city_lookup(df, validation_raw)
    dec_out = predict_december(pipeline, weight_median, city_lookup)
    dec_out.to_csv(ROOT / "december_predictions.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
