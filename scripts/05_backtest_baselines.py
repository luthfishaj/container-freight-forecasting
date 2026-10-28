# ---------------------------------------------------------------
# Script: 05_backtest_baselines.py
# Purpose: Rolling-origin backtest of five baseline forecasts on every
#          lane and horizon, and the resulting leaderboard.
# Input:   data/processed/panel.parquet
# Output:  data/processed/forecasts_baselines.csv  every forecast
#          (git-ignored: it contains index values)
#          reports/leaderboard.csv  metrics per model, lane, horizon
#          and period (public: aggregates only)
#          MLflow experiment "backtest": one parent run "baselines"
#          with one child run per model (hff/tracking.py)
#
# Baselines, all fitted on the log rate y_known up to the origin t:
#   naive           y_hat[t+h | t] = y[t]               (no change)
#   seasonal_naive  y_hat[t+h | t] = y[t+h-52]          (same week last year)
#   drift           y_hat[t+h | t] = y[t] + h * average weekly change
#                   over the whole history
#   ets             exponential smoothing with a damped additive trend
#   arima           ARIMA(1,1,0): weekly changes depend on last week's
#                   change
# ---------------------------------------------------------------

import warnings

import numpy as np
import pandas as pd

from hff.backtest import first_origin_position, summarise_forecasts, training_scale
from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel
from hff.models import (
    forecast_arima,
    forecast_drift,
    forecast_ets,
    forecast_naive,
    forecast_seasonal_naive,
)
from hff.tracking import log_backtest_run

config = load_config()
horizons = config["horizons"]
max_horizon = max(horizons)
processed_folder = PROJECT_ROOT / config["paths"]["processed_data"]
reports_folder = PROJECT_ROOT / "reports"

# statsmodels warns about convergence on short or flat histories; the
# script counts failed fits instead of printing every warning.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------
# 1. Baseline models (defined in hff.models)
# ---------------------------------------------------------------
def forecast_arima_baseline(history: np.ndarray, steps: int) -> np.ndarray:
    return forecast_arima(history, steps, config["arima_order"])


MODELS = {
    "naive": forecast_naive,
    "seasonal_naive": forecast_seasonal_naive,
    "drift": forecast_drift,
    "ets": forecast_ets,
    "arima": forecast_arima_baseline,
}

# ---------------------------------------------------------------
# 2. Rolling-origin loop
# ---------------------------------------------------------------
panel = load_panel(config)
forecast_rows = []
failed_fits = {model_name: 0 for model_name in MODELS}

for lane_id in panel["lane_id"].unique():
    lane = panel[panel["lane_id"] == lane_id].reset_index(drop=True)
    first_known = int(np.argmax(lane["y_known"].notna().to_numpy()))
    first_origin = first_origin_position(
        lane["y_known"], config["backtest_warm_up_weeks"]
    )
    last_origin = len(lane) - 2  # at least one week must follow the origin
    print(f"{lane_id}: origins {first_origin} to {last_origin}")

    for origin in range(first_origin, last_origin + 1):
        history = lane.loc[first_known:origin, "y_known"].to_numpy()
        scale = training_scale(history)

        for model_name, forecast_function in MODELS.items():
            try:
                path = forecast_function(history, max_horizon)
            except Exception:
                # A failed fit falls back to "no change" and is counted.
                failed_fits[model_name] += 1
                path = forecast_naive(history, max_horizon)

            for horizon in horizons:
                target = origin + horizon
                if target >= len(lane) or pd.isna(lane.loc[target, "y"]):
                    continue  # no published value to compare with
                forecast_rows.append(
                    {
                        "model": model_name,
                        "lane_id": lane_id,
                        "origin_date": lane.loc[origin, "week_date"],
                        "horizon": horizon,
                        "target_date": lane.loc[target, "week_date"],
                        "y_last": history[-1],
                        "y_forecast": float(path[horizon - 1]),
                        "y_actual": lane.loc[target, "y"],
                        "scale": scale,
                        "period": lane.loc[target, "period"],
                    }
                )

forecasts = pd.DataFrame(forecast_rows)
print(f"{len(forecasts)} forecasts; failed fits per model: {failed_fits}")

# ---------------------------------------------------------------
# 3. Leaderboard
# ---------------------------------------------------------------
leaderboard = summarise_forecasts(forecasts)
forecasts.to_csv(processed_folder / "forecasts_baselines.csv", index=False)
leaderboard.to_csv(reports_folder / "leaderboard.csv", index=False)

# Short view: MASE per model and horizon on the main lane, all periods.
main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
view = leaderboard[
    (leaderboard["lane_id"] == main_lane) & (leaderboard["period"] == "all")
]
print(f"MASE on {main_lane}, all periods (lower is better):")
print(view.pivot(index="model", columns="horizon", values="mase").round(2).to_string())
print("Saved reports/leaderboard.csv")

# ---------------------------------------------------------------
# 4. MLflow: one parent run, one child run per baseline
# ---------------------------------------------------------------
MODEL_SETTINGS = {
    "naive": {},
    "seasonal_naive": {"season_weeks": 52},
    "drift": {},
    "ets": {"trend": "additive, damped"},
    "arima": {"order": config["arima_order"]},
}
failed_fit_metrics = {}
for model_name, count in failed_fits.items():
    failed_fit_metrics[model_name] = {"failed_fits": count}
log_backtest_run(
    "baselines",
    config,
    MODEL_SETTINGS,
    leaderboard,
    [reports_folder / "leaderboard.csv"],
    child_extra_metrics=failed_fit_metrics,
)
