"""Generates the exploratory-analysis and model-diagnostic figures used in
the written report. Run after train.py (it reuses reports/model.joblib).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import ALL_FEATURES, clean_and_engineer, fit_weight_median

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLOR = "#064A56"


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "data" / "train-test.csv")
    df["date"] = pd.to_datetime(df["date"])

    # 1. posted_rate distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["posted_rate"], bins=80, color=COLOR)
    ax.set_title("Distribution of posted_rate")
    ax.set_xlabel("posted_rate ($)")
    ax.set_ylabel("count")
    savefig(fig, "rate_distribution.png")

    # 2. rate vs distance
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["distance"], df["posted_rate"], s=4, alpha=0.25, color=COLOR)
    ax.set_title("posted_rate vs distance (corr = {:.2f})".format(df["distance"].corr(df["posted_rate"])))
    ax.set_xlabel("distance (mi)")
    ax.set_ylabel("posted_rate ($)")
    savefig(fig, "rate_vs_distance.png")

    # 3. monthly seasonality (rate per mile)
    d = df.copy()
    d["rate_per_mile"] = d["posted_rate"] / d["distance"]
    d["month"] = d["date"].dt.to_period("M").astype(str)
    monthly = d.groupby("month")["rate_per_mile"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color=COLOR)
    ax.set_title("Average rate-per-mile by month (2025)")
    ax.set_ylabel("$ / mile")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, "monthly_seasonality.png")

    # 4. day-of-week seasonality
    d["dow"] = d["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = d.groupby("dow")["rate_per_mile"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dow.index, dow.values, color=COLOR)
    ax.set_title("Average rate-per-mile by day of week")
    ax.set_ylabel("$ / mile")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, "dow_seasonality.png")

    # 5. weight sign-error before/after cleaning
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["weight"].dropna(), bins=80, color=COLOR)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("weight before cleaning (negative values = sign-flip errors)")
    ax.set_xlabel("weight (lb)")
    savefig(fig, "weight_before_cleaning.png")

    # 6. holdout: actual vs predicted + residuals
    cutoff = pd.Timestamp("2025-09-01")
    train_raw = df[df["date"] < cutoff]
    holdout_raw = df[df["date"] >= cutoff]
    weight_median = fit_weight_median(train_raw)
    train = clean_and_engineer(train_raw, weight_median)
    holdout = clean_and_engineer(holdout_raw, weight_median)

    from src.model import build_pipeline

    pipeline = build_pipeline()
    pipeline.fit(train[ALL_FEATURES], train["posted_rate"])
    pred = pipeline.predict(holdout[ALL_FEATURES])
    actual = holdout["posted_rate"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = [0, max(actual.max(), pred.max()) * 1.02]
    ax.plot(lim, lim, color="red", linewidth=1, linestyle="--")
    ax.scatter(actual, pred, s=5, alpha=0.25, color=COLOR)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual posted_rate ($)")
    ax.set_ylabel("predicted rate ($)")
    ax.set_title("Holdout (Sep-Oct 2025): actual vs predicted")
    savefig(fig, "actual_vs_predicted.png")

    residual = pred - actual
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=80, color=COLOR)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Holdout residuals (predicted - actual)")
    savefig(fig, "residuals.png")

    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    print(f"Holdout MAE={mae:.2f} RMSE={rmse:.2f}")

    # 7. feature importance (permutation on the fitted holdout pipeline)
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, holdout[ALL_FEATURES], actual, n_repeats=8, random_state=42, n_jobs=-1
    )
    order_idx = np.argsort(result.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(
        np.array(ALL_FEATURES)[order_idx],
        result.importances_mean[order_idx],
        xerr=result.importances_std[order_idx],
        color=COLOR,
    )
    ax.set_title("Permutation importance (holdout, drop in R2)")
    savefig(fig, "feature_importance.png")


if __name__ == "__main__":
    main()
