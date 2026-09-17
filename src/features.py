"""Shared data cleaning and feature engineering for the freight rate model.

Used identically for training, validation-set scoring, and the December
chart inputs so there is no train/serve skew between the three pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns fed into the model. Deliberately excludes market_index and
# quote_signal: they are not present in december-chart-inputs.csv, and
# exploratory analysis showed they carry almost no linear or tree-importance
# signal for posted_rate (see reports/eda_findings.md). Keeping them out
# gives one consistent pipeline for both prediction targets.
NUMERIC_FEATURES = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["equipment"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_weight(weight: pd.Series) -> pd.Series:
    """Fix sign-flip data entry errors and return an absolute-value series."""
    return weight.abs()


def fit_weight_median(train_df: pd.DataFrame) -> float:
    """Median weight computed on cleaned training weights (fit on train only)."""
    return clean_weight(train_df["weight"]).median()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date = pd.to_datetime(out["date"])
    month = date.dt.month
    doy = date.dt.dayofyear
    dow = date.dt.dayofweek

    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out


def build_city_lookup(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Every city seen anywhere always carries the same lat/lon (checked
    during EDA), so we can back-fill coordinates for datasets that only give
    city names, such as december-chart-inputs.csv."""
    frames = []
    for df in dfs:
        if {"pickup", "pickup_lat", "pickup_lon"}.issubset(df.columns):
            frames.append(
                df[["pickup", "pickup_lat", "pickup_lon"]].rename(
                    columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
                )
            )
        if {"delivery", "delivery_lat", "delivery_lon"}.issubset(df.columns):
            frames.append(
                df[["delivery", "delivery_lat", "delivery_lon"]].rename(
                    columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
                )
            )
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="city").set_index("city")


def attach_geo(df: pd.DataFrame, city_lookup: pd.DataFrame) -> pd.DataFrame:
    """Add pickup/delivery lat & lon columns when a dataset (e.g. the
    December chart inputs) only has city names."""
    if {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}.issubset(df.columns):
        return df
    out = df.copy()
    out["pickup_lat"] = out["pickup"].map(city_lookup["lat"])
    out["pickup_lon"] = out["pickup"].map(city_lookup["lon"])
    out["delivery_lat"] = out["delivery"].map(city_lookup["lat"])
    out["delivery_lon"] = out["delivery"].map(city_lookup["lon"])
    if out[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].isna().any().any():
        missing = set(out.loc[out["pickup_lat"].isna(), "pickup"]) | set(
            out.loc[out["delivery_lat"].isna(), "delivery"]
        )
        raise ValueError(f"No known coordinates for cities: {sorted(missing)}")
    return out


def clean_and_engineer(df: pd.DataFrame, weight_median: float) -> pd.DataFrame:
    """Apply cleaning + feature engineering. `weight_median` must come from
    fit_weight_median(train_df) so validation/December data never leak
    their own statistics into the pipeline."""
    out = df.copy()
    out["weight"] = clean_weight(out["weight"])
    out["weight"] = out["weight"].fillna(weight_median)
    out = add_date_features(out)
    return out


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the final feature columns for the model."""
    return df[ALL_FEATURES]
