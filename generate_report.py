"""Builds reports/Freight_Rate_Report.pdf from the figures in reports/figures
and the metrics in reports/holdout_metrics.json. Run after train.py, eda.py
and score.py have all produced their outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
CHART = ROOT / "scorer_results" / "candidate_december.png"
METRICS = json.loads((ROOT / "reports" / "holdout_metrics.json").read_text())

DARK = (6, 74, 86)
GREY = (70, 70, 70)


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, "Freight Rate Prediction - Validation & Modeling Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R")
        self.ln(10)

    def h1(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*DARK)
        self.multi_cell(0, 9, text)
        self.ln(1)

    def h2(self, text):
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 8, text)
        self.ln(1)

    def body(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5.6, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(20, 20, 20)
        self.set_x(self.l_margin + 4)
        self.multi_cell(0, 5.6, f"-  {text}")

    def figure(self, path, caption, width=170):
        self.ln(1)
        x = (210 - width) / 2
        self.image(str(path), x=x, w=width)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*GREY)
        self.multi_cell(0, 5, caption, align="C")
        self.ln(6)


def main():
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 16, 20)

    # ---- Title page ----
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 12, "Freight Rate Prediction Challenge", align="C")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 8, "Validation Approach, Modeling Report & December Rate Forecast", align="C")
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.multi_cell(0, 6, "Candidate submission - Machine Learning Engineer Assessment", align="C")

    # ---- 1. Data overview ----
    pdf.add_page()
    pdf.h1("1. Data Overview & Key Findings")
    pdf.body(
        "train-test.csv contains 48,000 labeled loads from 2025-01-01 through 2025-10-31 across "
        "64 US cities, 3 equipment types (Dry Van, Reefer, Flatbed) and a target of posted_rate ($). "
        "validation.csv (12,000 loads, needing predictions) spans 2025-11-01 through 2025-12-31 and "
        "includes 8 cities never seen in training (Knoxville, San Diego, Norfolk, Jackson, Allentown, "
        "Charlotte, Chicago, Laredo), which shaped the feature design below. "
        "december-chart-inputs.csv holds 31 rows, one per December day, with pickup, delivery, "
        "distance, equipment and weight all fixed - only the date changes."
    )
    pdf.body("Key relationships found during EDA:")
    pdf.bullet("distance is by far the strongest predictor (Pearson r = 0.91 with posted_rate).")
    pdf.bullet(
        "Equipment type has a smaller but consistent effect on rate per mile: Reefer > Flatbed > Dry Van."
    )
    pdf.bullet(
        "Rate per mile shows genuine seasonality: it rises from January into a June peak "
        "(~$2.33/mi) and eases back down through the fall (~$2.19-2.24/mi), a plausible "
        "summer-freight-demand pattern."
    )
    pdf.bullet(
        "market_index and quote_signal have very weak linear correlation with posted_rate "
        "(|r| < 0.07) and were confirmed low-importance in model testing - see Section 3."
    )
    pdf.figure(FIG / "rate_vs_distance.png", "Figure 1. posted_rate vs distance", width=140)
    pdf.figure(FIG / "monthly_seasonality.png", "Figure 2. Average rate-per-mile by month", width=140)

    # ---- 2. Data quality issues ----
    pdf.add_page()
    pdf.h1("2. Data Quality Issues & How They Were Addressed")
    pdf.h2("2.1 Sign-flip errors in weight")
    pdf.body(
        "292 of 48,000 rows (0.6%) had a negative weight. The negated values' distribution is "
        "statistically identical to the positive-weight distribution (same mean, std and range), "
        "confirming a sign-entry error rather than a distinct population. Fix: take the absolute "
        "value (src/features.py: clean_weight)."
    )
    pdf.figure(FIG / "weight_before_cleaning.png", "Figure 3. Raw weight distribution before cleaning", width=130)
    pdf.h2("2.2 Missing values")
    pdf.body(
        "weight was missing for 300 training rows (0.6%) and 165 validation rows; imputed with the "
        "median weight computed on the training split only, so no information leaks across the "
        "train/holdout boundary or into validation. market_index was missing for 374 training rows "
        "(0.8%) and 249 validation rows, but since this feature was excluded from the model "
        "(Section 3), no imputation was needed for it."
    )
    pdf.h2("2.3 Unseen cities in validation.csv")
    pdf.body(
        "8 of the 72 cities in validation.csv never appear in training. A model that encodes pickup/"
        "delivery as plain categorical IDs (one-hot or target encoding) has no way to score these "
        "rows sensibly. Instead, pickup/delivery latitude & longitude are used as continuous "
        "features, which generalize smoothly to new cities. A stress test (see Section 3) confirmed "
        "this: predictions on routes touching held-out cities were only ~15% worse than for fully-"
        "seen routes, rather than failing outright."
    )
    pdf.h2("2.4 Distance is not a fixed lane property")
    pdf.body(
        "3,944 of 4,014 pickup-delivery city pairs have more than one distinct distance value across "
        "loads, i.e. distance varies per shipment (routing, detours) rather than being a lookup "
        "constant. This is expected and was used per-row, not averaged per lane."
    )
    pdf.h2("2.5 Unexplained high-rate outliers")
    pdf.body(
        "~0.7% of loads have a rate-per-mile more than double the typical range ($10-14/mi vs. a "
        "~$2/mi median), with no correlation to equipment, month, market_index, quote_signal or "
        "negative-weight flags that were tested. These look like genuine spot-market rate spikes "
        "(e.g. capacity crunches) that are not explainable from the given features, so they were "
        "kept in training rather than dropped as errors, and error metrics were evaluated with both "
        "MAE/MAPE (robust) and RMSE (sensitive to this tail) for transparency."
    )

    # ---- 3. Feature engineering & model ----
    pdf.add_page()
    pdf.h1("3. Feature Engineering & Model Selection")
    pdf.h2("3.1 Features used")
    pdf.body(
        "distance, weight (cleaned), pickup/delivery lat & lon, equipment (one-hot), and date "
        "decomposed into cyclical sin/cos encodings for month, day-of-year and day-of-week plus an "
        "is_weekend flag. market_index and quote_signal were deliberately excluded: they are absent "
        "from december-chart-inputs.csv, and an ablation showed they change holdout MAE by less than "
        "1%, so dropping them keeps one consistent pipeline for both the validation and December "
        "predictions instead of maintaining a second model or an imputation scheme for missing "
        "market signals in December."
    )
    pdf.h2("3.2 Train/validation split")
    pdf.body(
        f"The real prediction task is forecasting dates the model has never trained on (validation.csv "
        "runs Nov-Dec 2025; the December chart needs Dec 2025), so a random K-fold split would "
        "overstate accuracy by letting the model interpolate between dates it has already seen on "
        "both sides. Instead, a time-based holdout was used: train on "
        f"{METRICS['n_train']:,} rows from 2025-01-01 to 2025-08-31, evaluate on "
        f"{METRICS['n_holdout']:,} rows from {METRICS['holdout_start']} to 2025-10-31 (the most "
        "recent ~2 months, unseen during training). This mirrors the extrapolation the final model "
        "must do into November and December. A separate stress test additionally re-trained the "
        "model with 8 cities fully removed to confirm generalization to unseen cities specifically "
        "(Section 2.3)."
    )
    pdf.h2("3.3 Model comparison")
    pdf.body(
        "Five models were compared on the same time-based holdout: Linear Regression, Random Forest, "
        "Gradient Boosting, XGBoost, and HistGradientBoostingRegressor. HistGradientBoostingRegressor "
        "with an absolute-error objective gave the best MAE and MAPE and was competitive on RMSE, "
        "and was selected as the final model (src/model.py). It also natively tolerates the mild "
        "outlier tail described in Section 2.5 better than a squared-error objective."
    )
    pdf.h2("3.4 Final holdout performance")
    pdf.bullet(f"MAE:  ${METRICS['mae']:.2f}")
    pdf.bullet(f"RMSE: ${METRICS['rmse']:.2f}  (inflated by the unexplained-outlier tail, Section 2.5)")
    pdf.bullet(f"MAPE: {METRICS['mape']*100:.2f}%")
    pdf.bullet(f"R2:   {METRICS['r2']:.3f}")
    pdf.figure(FIG / "actual_vs_predicted.png", "Figure 4. Holdout actual vs. predicted rate", width=110)

    # ---- 4. Feature importance ----
    pdf.add_page()
    pdf.h1("4. Feature Importance")
    pdf.body(
        "Permutation importance on the holdout set (drop in R2 when a feature is shuffled) confirms "
        "distance dominates the model, with equipment and weight contributing a small but non-zero "
        "amount. The geographic and date-cyclical features individually contribute very little once "
        "distance/equipment/weight are known - consistent with the weak raw correlations found in "
        "Section 1 - but they are what let the model generalize to unseen cities and produce a "
        "sensible December date trend (Section 5)."
    )
    pdf.figure(FIG / "feature_importance.png", "Figure 5. Permutation importance by feature", width=130)

    # ---- 5. December chart ----
    pdf.add_page()
    pdf.h1("5. December 2025 Prediction Chart")
    pdf.body(
        "The chart below is produced by the provided score.py from december_predictions.csv (fixed "
        "route: Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb; only the date varies)."
    )
    pdf.figure(CHART, "Figure 6. score.py output: scorer_results/candidate_december.png", width=175)
    pdf.body(
        "Interpretation & caveat: because date-based features carry very little importance in the "
        "fitted model (Section 4), the predicted range across all 31 December days is narrow "
        "(about $772-$790, a ~2.3% spread) and shows a mild weekday-higher / weekend-lower pattern. "
        "This should be read as a small residual seasonal effect layered on a rate that is "
        "overwhelmingly set by distance, equipment and weight (all fixed for this route) - not as "
        "a strong, high-confidence seasonal forecast. Tree-based models also cannot extrapolate a "
        "linear trend beyond the training range; cyclical sin/cos date encodings were used "
        "specifically so that November/December values fall inside the numeric range already seen "
        "in training (see src/features.py), avoiding wild extrapolation, at the cost of not being "
        "able to project a trend that continues indefinitely upward or downward past October."
    )

    # ---- 6. Reproducing ----
    pdf.add_page()
    pdf.h1("6. Reproducing These Results")
    pdf.body("From the repository root:")
    pdf.set_font("Courier", "", 9.5)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(
        0,
        5.2,
        "python -m pip install -r requirements.txt\n"
        "python train.py      # cleans data, validates, fits final model, writes\n"
        "                      # validation_predictions.csv and december_predictions.csv\n"
        "python eda.py         # regenerates reports/figures used in this report\n"
        "python score.py --predictions validation_predictions.csv \\\n"
        "                 --december-predictions december_predictions.csv\n"
        "python generate_report.py   # rebuilds this PDF",
    )
    pdf.set_font("Helvetica", "", 10.5)
    pdf.ln(3)
    pdf.body(
        "See README.md for a description of the repository layout (src/features.py, src/model.py, "
        "train.py, eda.py, score.py) and the full run instructions."
    )

    out = ROOT / "reports" / "Freight_Rate_Report.pdf"
    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
