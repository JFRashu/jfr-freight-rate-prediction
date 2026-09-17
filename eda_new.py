"""
eda_new.py — EDA figures + multi-model comparison
Run: python eda_new.py
Outputs all figures to reports/figures/ and prints model comparison table.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
FIG_DIR   = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TEAL = "#064A56"
RED  = "#E74C3C"

# ── load & clean ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "train-test.csv")
df["date"] = pd.to_datetime(df["date"])
df["weight"] = df["weight"].abs()                        # fix sign-flip errors
df["weight"] = df["weight"].fillna(df["weight"].median())
df["rate_per_mile"] = df["posted_rate"] / df["distance"]
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["day_name"] = df["date"].dt.day_name()
df["equip_enc"] = df["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2})

# ── feature set (mirrors src/features.py) ─────────────────────────────────────
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["weight"] = d["weight"].abs().fillna(d["weight"].abs().median())
    doy = d["date"].dt.dayofyear
    dow = d["date"].dt.dayofweek
    mon = d["date"].dt.month
    d["doy_sin"]     = np.sin(2 * np.pi * doy / 365)
    d["doy_cos"]     = np.cos(2 * np.pi * doy / 365)
    d["dow_sin"]     = np.sin(2 * np.pi * dow / 7)
    d["dow_cos"]     = np.cos(2 * np.pi * dow / 7)
    d["month_sin"]   = np.sin(2 * np.pi * mon / 12)
    d["month_cos"]   = np.cos(2 * np.pi * mon / 12)
    d["is_weekend"]  = (dow >= 5).astype(int)
    d["equip_enc"]   = d["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2})
    return d

FEATURES = [
    "distance", "weight",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "equip_enc",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend",
]

CUTOFF = pd.Timestamp("2025-09-01")
df_f = add_features(df)
train_f = df_f[df_f["date"] < CUTOFF]
hold_f  = df_f[df_f["date"] >= CUTOFF]
X_tr, y_tr = train_f[FEATURES], train_f["posted_rate"]
X_ho, y_ho = hold_f[FEATURES],  hold_f["posted_rate"]

print(f"Train: {len(X_tr):,}  Holdout: {len(X_ho):,}")

# ══════════════════════════════════════════════════════════════════════════════
# EDA FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# 1. posted_rate vs distance
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["distance"], df["posted_rate"], alpha=0.15, s=4, color=TEAL)
ax.set_xlabel("distance (mi)")
ax.set_ylabel("posted_rate ($)")
ax.set_title(f"posted_rate vs distance (corr = {df['distance'].corr(df['posted_rate']):.2f})")
fig.tight_layout()
fig.savefig(FIG_DIR / "rate_vs_distance.png", dpi=150)
plt.close(fig)
print("saved: rate_vs_distance.png")

# 2. Rate distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["posted_rate"], bins=80, color=TEAL, edgecolor="none")
ax.set_xlabel("posted_rate ($)")
ax.set_ylabel("count")
ax.set_title("Distribution of posted_rate")
fig.tight_layout()
fig.savefig(FIG_DIR / "rate_distribution.png", dpi=150)
plt.close(fig)
print("saved: rate_distribution.png")

# 3. Weight before cleaning
raw_weight = pd.read_csv(DATA_DIR / "train-test.csv")["weight"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(raw_weight.dropna(), bins=100, color=TEAL, edgecolor="none")
ax.axvline(0, color=RED, linestyle="--", linewidth=1.5)
ax.set_xlabel("weight (lb)")
ax.set_ylabel("count")
ax.set_title("weight before cleaning (negative values = sign-flip errors)")
fig.tight_layout()
fig.savefig(FIG_DIR / "weight_before_cleaning.png", dpi=150)
plt.close(fig)
print("saved: weight_before_cleaning.png")

# 4. Monthly seasonality
monthly = df.groupby(df["date"].dt.to_period("M"))["rate_per_mile"].mean()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(monthly.index.astype(str), monthly.values, marker="o", color=TEAL, linewidth=2)
ax.set_xlabel("")
ax.set_ylabel("$ / mile")
ax.set_title("Average rate-per-mile by month (2025)")
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(FIG_DIR / "monthly_seasonality.png", dpi=150)
plt.close(fig)
print("saved: monthly_seasonality.png")

# 5. Day-of-week seasonality
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_rpm = df.groupby("day_name")["rate_per_mile"].mean().reindex(dow_order)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(dow_order, dow_rpm.values, color=TEAL)
ax.set_ylabel("$ / mile")
ax.set_title("Average rate-per-mile by day of week")
fig.tight_layout()
fig.savefig(FIG_DIR / "dow_seasonality.png", dpi=150)
plt.close(fig)
print("saved: dow_seasonality.png")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Training models on holdout split ──")

models = {
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
    ),
    "HistGradientBoosting\n(squared loss)": HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6,
        random_state=42, loss="squared_error"
    ),
    "HistGradientBoosting\n(absolute loss) ✓": HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6,
        random_state=42, loss="absolute_error"
    ),
}

results = {}
for name, model in models.items():
    print(f"  fitting {name.strip()} ...", end=" ", flush=True)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ho)
    mae  = mean_absolute_error(y_ho, pred)
    rmse = mean_squared_error(y_ho, pred) ** 0.5
    mape = mean_absolute_percentage_error(y_ho, pred) * 100
    r2   = r2_score(y_ho, pred)
    results[name] = dict(mae=mae, rmse=rmse, mape=mape, r2=r2, pred=pred)
    print(f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.2f}%  R²={r2:.4f}")

# ── model comparison bar chart ─────────────────────────────────────────────────
names_clean = [n.replace("\n", "\n") for n in results]
maes  = [results[n]["mae"]  for n in results]
mapes = [results[n]["mape"] for n in results]
r2s   = [results[n]["r2"]   for n in results]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = [TEAL if "✓" not in n else "#E67E22" for n in results]

for ax, vals, title, fmt in zip(
    axes,
    [maes, mapes, r2s],
    ["MAE ($) — lower is better",
     "MAPE (%) — lower is better",
     "R² — higher is better"],
    ["${:.0f}", "{:.2f}%", "{:.4f}"],
):
    bars = ax.barh(names_clean, vals, color=colors)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Model Comparison — Time-based Holdout (Sep–Oct 2025)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved: model_comparison.png")

# ── use best model (HistGBR absolute) for remaining figures ────────────────────
best_name = "HistGradientBoosting\n(absolute loss) ✓"
best_pred = results[best_name]["pred"]

# 6. Actual vs predicted
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_ho, best_pred, alpha=0.15, s=4, color=TEAL)
lim = max(y_ho.max(), best_pred.max())
ax.plot([0, lim], [0, lim], "--", color=RED, linewidth=1.2)
ax.set_xlabel("actual posted_rate ($)")
ax.set_ylabel("predicted rate ($)")
ax.set_title("Holdout (Sep-Oct 2025): actual vs predicted")
fig.tight_layout()
fig.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=150)
plt.close(fig)
print("saved: actual_vs_predicted.png")

# 7. Residuals
residuals = best_pred - y_ho.to_numpy()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=120, color=TEAL, edgecolor="none")
ax.axvline(0, color=RED, linestyle="--", linewidth=1.5)
ax.set_xlabel("residual ($)")
ax.set_ylabel("count")
ax.set_title("Holdout residuals (predicted - actual)")
fig.tight_layout()
fig.savefig(FIG_DIR / "residuals.png", dpi=150)
plt.close(fig)
print("saved: residuals.png")

# 8. Feature importance (permutation — HistGBR absolute)
from sklearn.inspection import permutation_importance
best_model = models[best_name]
pi = permutation_importance(best_model, X_ho, y_ho, n_repeats=5,
                             scoring="r2", random_state=42, n_jobs=-1)
sorted_idx = pi.importances_mean.argsort()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([FEATURES[i] for i in sorted_idx],
        pi.importances_mean[sorted_idx],
        xerr=pi.importances_std[sorted_idx],
        color=TEAL)
ax.set_title("Permutation importance (holdout, drop in R²)")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
plt.close(fig)
print("saved: feature_importance.png")

# ── save comparison JSON ───────────────────────────────────────────────────────
comparison = {
    k.replace("\n",""): {kk: vv for kk, vv in v.items() if kk != "pred"}
    for k, v in results.items()
}
(ROOT / "reports" / "model_comparison.json").write_text(
    json.dumps(comparison, indent=2)
)

# ── print summary table ────────────────────────────────────────────────────────
print("\n═══════════════════════════════════════════════════════════════════")
print(f"{'Model':<42} {'MAE':>8} {'RMSE':>9} {'MAPE':>8} {'R²':>8}")
print("─" * 77)
for name, m in results.items():
    flag = " ← SELECTED" if "✓" in name else ""
    print(f"{name.replace(chr(10),' '):<42} ${m['mae']:>7.1f} ${m['rmse']:>8.1f} {m['mape']:>7.2f}% {m['r2']:>8.4f}{flag}")
print("═══════════════════════════════════════════════════════════════════")
print("\nAll figures saved to reports/figures/")