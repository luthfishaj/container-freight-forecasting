# ---------------------------------------------------------------
# Script: 10_pricing_policy.py
# Purpose: Illustration of business value, not a pricing engine. Each
#          week t, a carrier on the main lane chooses between:
#            lock  fix today's rate r[t] for the next 4 weeks
#            spot  earn each week's spot rate r[t+1], ..., r[t+4]
#          Locking pays off when rates fall (D-010, carrier's side).
# Input:   data/processed/panel.parquet
#          data/processed/forecasts_baselines.csv, forecasts_models.csv
# Output:  reports/pricing_policy.csv   value of each policy by period
#          reports/figures/pricing_policy.png
#          MLflow experiment "business", run "pricing"
#
# Value of locking in week t, in percent of today's rate:
#   value_lock[t] = 100 * (r[t] - mean(r[t+1..t+4])) / r[t]
# Policies compared, all measured against "always spot" (value 0):
#   always_lock          lock every week
#   forecast (<model>)   lock when the model's 4-week median forecast
#                        is below today's rate
#   perfect_foresight    lock exactly when rates will fall (upper bound)
# Simplifications, stated in the report: every week is treated as an
# independent decision (the 4-week windows overlap); volumes, costs,
# customer reactions and contract terms are ignored.
# ---------------------------------------------------------------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from hff.charts import DRIVER_INK, TEXT_MUTED, apply_style
from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel
from hff.tracking import log_finite_metrics, start_tracking

config = load_config()
lock_weeks = config["pricing_lock_weeks"]
processed_folder = PROJECT_ROOT / config["paths"]["processed_data"]
reports_folder = PROJECT_ROOT / "reports"
figures_folder = PROJECT_ROOT / config["paths"]["figures"]
apply_style()
FORECAST_MODELS = ["drift", "arima", "sarimax", "lightgbm"]

# ---------------------------------------------------------------
# 1. Value of locking, week by week, on the main lane
# ---------------------------------------------------------------
main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
panel = load_panel(config)
lane = panel[panel["lane_id"] == main_lane].reset_index(drop=True)
rate_known = np.exp(lane["y_known"])
rate_actual = np.exp(lane["y"])

decisions = []
for week in range(len(lane) - lock_weeks):
    next_rates = rate_actual.iloc[week + 1 : week + 1 + lock_weeks]
    if next_rates.isna().any() or pd.isna(rate_known.iloc[week]):
        continue  # the outcome of the decision cannot be measured
    spot_average = next_rates.mean()
    decisions.append(
        {
            "origin_date": lane.loc[week, "week_date"],
            "period": lane.loc[week + lock_weeks, "period"],
            "value_lock_pct": 100
            * (rate_known.iloc[week] - spot_average)
            / rate_known.iloc[week],
        }
    )
decisions = pd.DataFrame(decisions)

# ---------------------------------------------------------------
# 2. Forecast-based decisions (4-week median forecasts)
# ---------------------------------------------------------------
forecasts = pd.concat(
    [
        pd.read_csv(processed_folder / "forecasts_baselines.csv"),
        pd.read_csv(processed_folder / "forecasts_models.csv"),
    ],
    ignore_index=True,
)
forecasts = forecasts[
    (forecasts["lane_id"] == main_lane)
    & (forecasts["horizon"] == lock_weeks)
    & forecasts["model"].isin(FORECAST_MODELS)
]
forecasts["origin_date"] = pd.to_datetime(forecasts["origin_date"])
forecasts["predicts_fall"] = forecasts["y_forecast"] < forecasts["y_last"]
signals = forecasts.pivot_table(
    index="origin_date", columns="model", values="predicts_fall"
)
decisions = decisions.merge(
    signals, left_on="origin_date", right_index=True, how="inner"
)

# Value of each policy in each week, relative to always staying on spot.
policies = {"always_lock": decisions["value_lock_pct"]}
for model_name in FORECAST_MODELS:
    lock_now = decisions[model_name].astype(bool)
    policies[f"forecast_{model_name}"] = decisions["value_lock_pct"].where(
        lock_now, 0.0
    )
policies["perfect_foresight"] = decisions["value_lock_pct"].clip(lower=0)

# ---------------------------------------------------------------
# 3. Results by period
# ---------------------------------------------------------------
result_rows = []
for period in ["all"] + sorted(decisions["period"].unique()):
    in_period = (
        decisions["period"] == period
        if period != "all"
        else decisions["period"].notna()
    )
    for policy_name, values in policies.items():
        locked = values[in_period] != 0
        result_rows.append(
            {
                "period": period,
                "policy": policy_name,
                "weeks": int(in_period.sum()),
                "share_of_weeks_locked": round(float(locked.mean()), 2),
                "mean_value_vs_spot_pct": round(float(values[in_period].mean()), 2),
            }
        )
results = pd.DataFrame(result_rows)
results.to_csv(reports_folder / "pricing_policy.csv", index=False)
print("Mean value per decision vs always spot, % of the rate:")
print(
    results.pivot(
        index="policy", columns="period", values="mean_value_vs_spot_pct"
    ).to_string()
)

# ---------------------------------------------------------------
# 4. Figure: mean value per policy, all periods
# ---------------------------------------------------------------
view = results[results["period"] == "all"].set_index("policy")["mean_value_vs_spot_pct"]
figure, axis = plt.subplots(figsize=(8, 3.6))
positions = range(len(view))
axis.barh(list(positions), view.to_numpy(), color=DRIVER_INK, height=0.6)
# Each value is written just outside the end of its bar: to the right
# of positive bars, to the left of negative ones.
for position, value in zip(positions, view.to_numpy()):
    if value >= 0:
        offset, alignment = 4, "left"
    else:
        offset, alignment = -4, "right"
    axis.annotate(
        f"{value:+.1f}%",
        (value, position),
        xytext=(offset, 0),
        textcoords="offset points",
        va="center",
        ha=alignment,
        fontsize=8,
    )
axis.margins(x=0.15)  # room for the labels at both ends
axis.set_yticks(list(positions), [name.replace("_", " ") for name in view.index])
axis.axvline(0, color=TEXT_MUTED, linewidth=0.8)
axis.set_xlabel("Mean value per weekly decision vs always spot, % of rate")
axis.set_title("Lock a 4-week rate or stay on spot: Middle East lane")
figure.savefig(figures_folder / "pricing_policy.png", dpi=150, bbox_inches="tight")
plt.close(figure)
print("Saved reports/pricing_policy.csv and reports/figures/pricing_policy.png")

# ---------------------------------------------------------------
# 5. MLflow: value of each policy per period, and the files
# ---------------------------------------------------------------
# Metric names: value_<policy>_<period>, mean value per weekly
# decision against always staying on spot, in percent of the rate.
start_tracking("business")
with mlflow.start_run(run_name="pricing"):
    mlflow.log_params({"lane": main_lane, "lock_weeks": lock_weeks})
    pricing_metrics = {}
    for row in results.itertuples():
        pricing_metrics[f"value_{row.policy}_{row.period}"] = row.mean_value_vs_spot_pct
    log_finite_metrics(pricing_metrics)
    mlflow.log_artifact(str(reports_folder / "pricing_policy.csv"))
    mlflow.log_artifact(str(figures_folder / "pricing_policy.png"))
print("Logged the run to MLflow (experiment: business)")
