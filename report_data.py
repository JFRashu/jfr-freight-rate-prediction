# Save this as report_data.py and run: python report_data.py
import pandas as pd
import numpy as np
import json

train = pd.read_csv("data/train-test.csv")
val   = pd.read_csv("data/validation.csv")
dec   = pd.read_csv("data/december-chart-inputs.csv")
dec_pred = pd.read_csv("december_predictions.csv")
val_pred = pd.read_csv("validation_predictions.csv")
metrics  = json.load(open("reports/holdout_metrics.json"))

train["date"] = pd.to_datetime(train["date"])
val["date"]   = pd.to_datetime(val["date"])

train_cities = set(train["pickup"]) | set(train["delivery"])
val_cities   = set(val["pickup"])   | set(val["delivery"])
unseen = val_cities - train_cities

train["weight_abs"] = train["weight"].abs()
neg_weight = (train["weight"] < 0).sum()
null_weight_train = train["weight"].isna().sum()
null_weight_val   = val["weight"].isna().sum()
null_mindex_train = train["market_index"].isna().sum()
null_mindex_val   = val["market_index"].isna().sum()

train["rate_per_mile"] = train["posted_rate"] / train["distance"]
monthly_rpm = train.groupby(train["date"].dt.to_period("M"))["rate_per_mile"].mean().round(3)

print("=== DATASET SUMMARY ===")
print(f"Train rows: {len(train)}, date range: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Val rows:   {len(val)},  date range: {val['date'].min().date()} to {val['date'].max().date()}")
print(f"Train cities (unique pickup+delivery): {len(train_cities)}")
print(f"Val cities (unique pickup+delivery):   {len(val_cities)}")
print(f"Unseen cities in val: {len(unseen)} -> {sorted(unseen)}")
print(f"Equipment types: {sorted(train['equipment'].unique())}")
print(f"Equipment counts:\n{train['equipment'].value_counts().to_string()}")

print("\n=== TARGET STATS ===")
print(train["posted_rate"].describe().round(2).to_string())
print(f"Outliers >$10k: {(train['posted_rate'] > 10000).sum()}")
print(f"Rate-per-mile >$5: {(train['rate_per_mile'] > 5).sum()} ({(train['rate_per_mile']>5).mean()*100:.2f}%)")

print("\n=== DATA QUALITY ===")
print(f"Negative weight rows: {neg_weight} ({neg_weight/len(train)*100:.2f}%)")
print(f"Null weight - train: {null_weight_train}, val: {null_weight_val}")
print(f"Null market_index - train: {null_mindex_train}, val: {null_mindex_val}")
print(f"Distance corr with posted_rate: {train['distance'].corr(train['posted_rate']):.4f}")
print(f"market_index corr: {train['market_index'].corr(train['posted_rate']):.4f}")
print(f"quote_signal corr: {train['quote_signal'].corr(train['posted_rate']):.4f}")

print("\n=== RATE PER MILE BY MONTH ===")
print(monthly_rpm.to_string())

print("\n=== RATE BY EQUIPMENT ===")
print(train.groupby("equipment")["rate_per_mile"].mean().round(3).to_string())

print("\n=== HOLDOUT METRICS ===")
print(json.dumps(metrics, indent=2))

print("\n=== DECEMBER PREDICTIONS ===")
print(dec_pred[["date","predicted_rate"]].to_string(index=False))
print(f"Min: {dec_pred['predicted_rate'].min():.2f}, Max: {dec_pred['predicted_rate'].max():.2f}")

print("\n=== VAL PREDICTIONS STATS ===")
print(val_pred["predicted_rate"].describe().round(2).to_string())