# ---------------------------------------------------------------
# Script: 11_weekly_forecast.py
# Purpose: Production step, run once a week after the data refresh
#          ("make weekly"): picks the champion model, forecasts every
#          lane 1 to 12 weeks ahead with an 80% interval, and saves a
#          dated snapshot (the live track record).
# Input:   data/processed/panel.parquet (rebuilt by "make weekly")
#          data/processed/forecasts_baselines.csv, forecasts_models.csv
# Output:  reports/forecasts/forecast_<week>.csv            public snapshot,
#                                         changes in % from the latest week
#          data/processed/forecast_snapshots/forecast_<week>.csv
#                                         same forecasts in log points,
#                                         kept locally for 12_monitor.py
#          MLflow experiment "weekly_forecast", one run per week
#
# Champion rule: among all backtested models, the one with the lowest
# MASE on the main lane, 4 weeks ahead, over the last 52 weeks of
# backtest origins. The naive forecast can be champion: a model is
# used only if it beat "no change" recently.
# Interval: champion forecast plus the 10% and 90% quantiles of the
# champion's own backtest errors, per lane and horizon, over the whole
# backtest (calm weeks and all three shocks). A 52-week window was
# tried and rejected: in late 2026 it held mostly Hormuz errors, which
# nearly all had the same sign, and gave intervals that excluded the
# median forecast.
# ---------------------------------------------------------------

import warnings

import mlflow
import numpy as np
import pandas as pd

from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel
from hff.metrics import mean_absolute_scaled_error
from hff.models import (
    add_model_features,
    fit_lightgbm,
    forecast_arima,
    forecast_drift,
    forecast_ets,
    forecast_naive,
    forecast_sarimax,
    forecast_seasonal_naive,
    lightgbm_features,
)
from hff.tracking import log_finite_metrics, start_tracking

config = load_config()
horizons = config["horizons"]
max_horizon = max(horizons)
processed_folder = PROJECT_ROOT / config["paths"]["processed_data"]
public_folder = PROJECT_ROOT / config["paths"]["forecasts"]
local_folder = processed_folder / "forecast_snapshots"
public_folder.mkdir(parents=True, exist_ok=True)
local_folder.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")
RECENT_WEEKS = 52
CHAMPION_HORIZON = 4
main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]

# ---------------------------------------------------------------
# 1. Champion: best recent backtest performance on the main lane
# ---------------------------------------------------------------
backtest = pd.concat(
    [
        pd.read_csv(processed_folder / "forecasts_baselines.csv"),
        pd.read_csv(processed_folder / "forecasts_models.csv"),
    ],
    ignore_index=True,
)
# Interval-only variants repeat the point forecasts of naive and
# lightgbm, so they are not separate candidates.
interval_only = backtest["model"].isin(["naive_empirical", "lightgbm_widened"])
backtest = backtest[~interval_only]
backtest["origin_date"] = pd.to_datetime(backtest["origin_date"])
recent_start = backtest["origin_date"].max() - pd.Timedelta(weeks=RECENT_WEEKS)
recent = backtest[backtest["origin_date"] > recent_start]

recent_main = recent[
    (recent["lane_id"] == main_lane) & (recent["horizon"] == CHAMPION_HORIZON)
]
recent_mase = {}
for model_name, group in recent_main.groupby("model"):
    recent_mase[model_name] = mean_absolute_scaled_error(
        group["y_actual"], group["y_forecast"], group["scale"]
    )
champion = min(recent_mase, key=recent_mase.get)
print(f"Recent MASE, {main_lane}, h = {CHAMPION_HORIZON}, last {RECENT_WEEKS} weeks:")
for model_name in sorted(recent_mase, key=recent_mase.get):
    print(f"  {model_name:15s} {recent_mase[model_name]:.3f}")
print(f"Champion: {champion}")

# ---------------------------------------------------------------
# 2. Point forecasts from the latest week
# ---------------------------------------------------------------
panel = load_panel(config)
panel = add_model_features(panel, config, horizons)
latest_week = panel["week_date"].max()


def champion_path(lane: pd.DataFrame) -> np.ndarray:
    """Forecast path y_hat[t+1..t+12 | t] of the champion for one lane."""
    history = lane[lane["y_known"].notna()]
    values = history["y_known"].to_numpy()
    if champion == "naive":
        return forecast_naive(values, max_horizon)
    if champion == "seasonal_naive":
        return forecast_seasonal_naive(values, max_horizon)
    if champion == "drift":
        return forecast_drift(values, max_horizon)
    if champion == "ets":
        return forecast_ets(values, max_horizon)
    if champion == "arima":
        return forecast_arima(values, max_horizon, config["arima_order"])
    if champion == "sarimax":
        _, mean_path, _, _ = forecast_sarimax(
            history, max_horizon, config["arima_order"]
        )
        return mean_path
    raise ValueError(f"No single-lane forecast for {champion}")


# LightGBM is global: one median model per horizon, trained on every
# row whose target was published before the latest week.
lightgbm_models = {}
if champion == "lightgbm":
    latest_position = int(panel["t"].max())
    for horizon in horizons:
        training_rows = panel[
            (panel["t"] + horizon <= latest_position - 1)
            & panel[f"target_{horizon}"].notna()
        ]
        lightgbm_models[horizon] = fit_lightgbm(training_rows, horizon, 0.5, config)

# ---------------------------------------------------------------
# 3. Intervals from the champion's backtest errors, and snapshots
# ---------------------------------------------------------------
champion_backtest = backtest[backtest["model"] == champion].copy()
champion_backtest["error"] = (
    champion_backtest["y_actual"] - champion_backtest["y_forecast"]
)

public_rows = []
local_rows = []
for lane_config in config["lanes"]:
    lane_id = lane_config["id"]
    lane = panel[panel["lane_id"] == lane_id].reset_index(drop=True)
    y_last = lane["y_known"].iloc[-1]
    if champion != "lightgbm":
        path = champion_path(lane)

    for horizon in horizons:
        if champion == "lightgbm":
            features = lane.loc[[len(lane) - 1], lightgbm_features(config)]
            median = y_last + lightgbm_models[horizon].predict(features)[0]
        else:
            median = float(path[horizon - 1])
        errors = champion_backtest[
            (champion_backtest["lane_id"] == lane_id)
            & (champion_backtest["horizon"] == horizon)
        ]["error"]
        low = median + errors.quantile(0.1)
        high = median + errors.quantile(0.9)
        target_week = (latest_week + pd.Timedelta(weeks=horizon)).date()
        local_rows.append(
            {
                "origin_week": latest_week.date(),
                "lane_id": lane_id,
                "horizon": horizon,
                "target_week": target_week,
                "model": champion,
                "y_last": y_last,
                "q10": low,
                "q50": median,
                "q90": high,
            }
        )
        public_rows.append(
            {
                "origin_week": latest_week.date(),
                "lane_id": lane_id,
                "role": lane_config["role"],
                "horizon": horizon,
                "target_week": target_week,
                "model": champion,
                "median_change_pct": round(100 * (np.exp(median - y_last) - 1), 1),
                "low_change_pct": round(100 * (np.exp(low - y_last) - 1), 1),
                "high_change_pct": round(100 * (np.exp(high - y_last) - 1), 1),
            }
        )

snapshot_name = f"forecast_{latest_week.date()}.csv"
pd.DataFrame(local_rows).to_csv(local_folder / snapshot_name, index=False)
public = pd.DataFrame(public_rows)
public.to_csv(public_folder / snapshot_name, index=False)

print(f"Forecasts from the week of {latest_week.date()} (change from that week, %):")
print(
    public[public["lane_id"] == main_lane][
        [
            "horizon",
            "target_week",
            "median_change_pct",
            "low_change_pct",
            "high_change_pct",
        ]
    ].to_string(index=False)
)
print(f"Saved reports/forecasts/{snapshot_name}")

# ---------------------------------------------------------------
# 4. MLflow: champion, recent scores and the main-lane forecast
# ---------------------------------------------------------------
start_tracking("weekly_forecast")
with mlflow.start_run(run_name=f"forecast {latest_week.date()}"):
    mlflow.set_tag("champion", champion)
    mlflow.log_params(
        {
            "origin_week": latest_week.date(),
            "champion": champion,
            "champion_rule": f"lowest MASE, {main_lane}, h={CHAMPION_HORIZON}, "
            f"last {RECENT_WEEKS} weeks",
        }
    )
    forecast_metrics = {}
    for model_name, score in recent_mase.items():
        forecast_metrics[f"recent_mase_{model_name}"] = score
    for row in public[public["lane_id"] == main_lane].itertuples():
        forecast_metrics[f"median_change_h{row.horizon}"] = row.median_change_pct
        forecast_metrics[f"low_change_h{row.horizon}"] = row.low_change_pct
        forecast_metrics[f"high_change_h{row.horizon}"] = row.high_change_pct
    log_finite_metrics(forecast_metrics)
    mlflow.log_artifact(str(public_folder / snapshot_name))
print("Logged the run to MLflow (experiment: weekly_forecast)")
