# ---------------------------------------------------------------
# Script: 06_backtest_models.py
# Purpose: Rolling-origin backtest of the driver-based models, with
#          10/50/90% quantile forecasts, compared with the baselines.
# Input:   data/processed/panel.parquet
#          data/processed/forecasts_baselines.csv (05_backtest_baselines.py)
# Output:  data/processed/forecasts_models.csv   every forecast (git-ignored)
#          reports/leaderboard.csv               all models, point metrics
#          reports/interval_metrics.csv          coverage, width, pinball
#          reports/feature_importance.csv        LightGBM, per horizon
#          reports/sarimax_coefficients.csv      driver effects per lane
#          reports/backtest_forecasts.csv        every forecast of the
#                                                main models, in percent
#                                                changes (dashboard)
#          MLflow experiment "backtest": parent run "driver_models",
#          one child run per model (hff/tracking.py)
#
# Models, all using only information known at the origin t:
#   sarimax          ARIMA(1,1,0) on y with the drivers as regressors
#                    x = log Brent, log GPR, log(1 + Gulf port calls).
#                    Future driver values are unknown at t, so they are
#                    held at their last known value over the horizon.
#                    The 80% interval comes from the model's own
#                    forecast distribution.
#   lightgbm         one model per horizon h and quantile alpha, trained
#                    on all lanes together (a lane code is a feature).
#                    Target: y[l, t+h] - y_known[l, t], the change over
#                    h weeks, so lanes at different levels share one
#                    model. Quantile alpha in {0.1, 0.5, 0.9}; the 0.5
#                    quantile is the point forecast.
#   lightgbm_widened the same forecasts, with the 10-90% interval widened
#                    from LightGBM's own past misses (section 3).
#   naive_empirical  benchmark interval: the naive forecast y[t] plus
#                    the 10% and 90% quantiles of its own past h-week
#                    errors on the same lane.
# ---------------------------------------------------------------

import warnings

import numpy as np
import pandas as pd

from hff.backtest import (
    first_origin_position,
    percent_change_table,
    summarise_forecasts,
    summarise_intervals,
    training_scale,
    widen_intervals,
)
from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel
from hff.models import (
    SARIMAX_DRIVERS,
    add_model_features,
    fit_lightgbm,
    forecast_sarimax,
    lightgbm_features,
)
from hff.tracking import log_backtest_run

config = load_config()
horizons = config["horizons"]
max_horizon = max(horizons)
quantile_levels = config["quantiles"]
processed_folder = PROJECT_ROOT / config["paths"]["processed_data"]
reports_folder = PROJECT_ROOT / "reports"
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# 1. Features (defined in hff.models)
# ---------------------------------------------------------------
panel = load_panel(config)
panel = add_model_features(panel, config, horizons)
lane_ids = sorted(panel["lane_id"].unique())
LIGHTGBM_FEATURES = lightgbm_features(config)

# ---------------------------------------------------------------
# 2. Rolling-origin loop
# ---------------------------------------------------------------
rows = []
lightgbm_models = {}
first_origin = first_origin_position(
    panel[panel["lane_id"] == lane_ids[0]]["y_known"], config["backtest_warm_up_weeks"]
)
last_origin = int(panel["t"].max()) - 1
print(f"Origins {first_origin} to {last_origin}, {len(lane_ids)} lanes")

for origin in range(first_origin, last_origin + 1):
    # LightGBM: refit every few weeks on rows whose target was already
    # published by the origin (target week t+h no later than origin-1).
    if (origin - first_origin) % config["retrain_every_weeks"] == 0:
        for horizon in horizons:
            known_targets = panel[
                (panel["t"] + horizon <= origin - 1)
                & panel[f"target_{horizon}"].notna()
            ]
            for alpha in quantile_levels:
                lightgbm_models[(horizon, alpha)] = fit_lightgbm(
                    known_targets, horizon, alpha, config
                )
        print(f"  origin {origin}: LightGBM refitted")

    for lane_id in lane_ids:
        lane = panel[panel["lane_id"] == lane_id].reset_index(drop=True)
        first_known = int(np.argmax(lane["y_known"].notna().to_numpy()))
        history = lane.loc[first_known:origin]
        y_last = history["y_known"].iloc[-1]
        scale = training_scale(history["y_known"].to_numpy())
        origin_row = lane.loc[[origin]]

        try:
            _, sarimax_mean, sarimax_low, sarimax_high = forecast_sarimax(
                history, max_horizon, config["arima_order"]
            )
        except Exception:
            sarimax_mean = None

        for horizon in horizons:
            target = origin + horizon
            if target >= len(lane) or pd.isna(lane.loc[target, "y"]):
                continue
            base = {
                "lane_id": lane_id,
                "origin_date": lane.loc[origin, "week_date"],
                "horizon": horizon,
                "target_date": lane.loc[target, "week_date"],
                "y_last": y_last,
                "y_actual": lane.loc[target, "y"],
                "scale": scale,
                "period": lane.loc[target, "period"],
            }

            # LightGBM quantiles, sorted so that q10 <= q50 <= q90.
            quantile_forecasts = []
            for alpha in quantile_levels:
                model = lightgbm_models[(horizon, alpha)]
                change = model.predict(origin_row[LIGHTGBM_FEATURES])[0]
                quantile_forecasts.append(y_last + change)
            quantile_forecasts.sort()
            rows.append(
                base
                | {
                    "model": "lightgbm",
                    "y_forecast": quantile_forecasts[1],
                    "q10": quantile_forecasts[0],
                    "q50": quantile_forecasts[1],
                    "q90": quantile_forecasts[2],
                }
            )

            if sarimax_mean is not None:
                rows.append(
                    base
                    | {
                        "model": "sarimax",
                        "y_forecast": sarimax_mean[horizon - 1],
                        "q10": sarimax_low[horizon - 1],
                        "q50": sarimax_mean[horizon - 1],
                        "q90": sarimax_high[horizon - 1],
                    }
                )

            # Benchmark interval from the naive forecast's past errors.
            past_errors = (
                (lane["y"].shift(-horizon) - lane["y_known"])
                .iloc[first_known : origin - horizon]
                .dropna()
            )
            if len(past_errors) >= 10:
                rows.append(
                    base
                    | {
                        "model": "naive_empirical",
                        "y_forecast": y_last,
                        "q10": y_last + past_errors.quantile(0.1),
                        "q50": y_last,
                        "q90": y_last + past_errors.quantile(0.9),
                    }
                )

forecasts = pd.DataFrame(rows)

# ---------------------------------------------------------------
# 3. Widened LightGBM intervals (conformal correction, D-030)
# ---------------------------------------------------------------
# The raw LightGBM quantiles are too narrow; each interval is widened
# from LightGBM's own past misses (hff.backtest.widen_intervals).
MIN_PAST_SCORES = 10
widened = widen_intervals(
    forecasts[forecasts["model"] == "lightgbm"], "lightgbm_widened", MIN_PAST_SCORES
)
forecasts = pd.concat([forecasts, widened], ignore_index=True)
forecasts.to_csv(processed_folder / "forecasts_models.csv", index=False)
print(f"{len(forecasts)} model forecasts")

# ---------------------------------------------------------------
# 4. Leaderboards
# ---------------------------------------------------------------
# The two interval-only variants repeat point forecasts already listed
# (naive and lightgbm), so they appear in the interval metrics only.
baselines = pd.read_csv(processed_folder / "forecasts_baselines.csv")
interval_only = forecasts["model"].isin(["naive_empirical", "lightgbm_widened"])
point_models = forecasts[~interval_only]
all_point = pd.concat([baselines, point_models], ignore_index=True)
leaderboard = summarise_forecasts(all_point)
leaderboard.to_csv(reports_folder / "leaderboard.csv", index=False)

interval_metrics = summarise_intervals(forecasts)
interval_metrics.to_csv(reports_folder / "interval_metrics.csv", index=False)

# ---------------------------------------------------------------
# 5. What drives the models
# ---------------------------------------------------------------
# LightGBM: importance (total gain) of each feature in the median
# models trained on all data, one column per horizon.
importance = pd.DataFrame({"feature": LIGHTGBM_FEATURES})
for horizon in horizons:
    training_rows = panel[panel[f"target_{horizon}"].notna()]
    model = fit_lightgbm(training_rows, horizon, 0.5, config)
    gains = model.booster_.feature_importance(importance_type="gain")
    importance[f"h{horizon}"] = 100 * gains / gains.sum()
importance = importance.sort_values(f"h{horizons[0]}", ascending=False)
importance.round(2).to_csv(reports_folder / "feature_importance.csv", index=False)

# SARIMAX: driver coefficients per lane, fitted on all data. A
# coefficient is the change in log rate per unit change of the
# driver's log, holding the ARIMA dynamics fixed.
coefficient_rows = []
for lane_id in lane_ids:
    lane = panel[panel["lane_id"] == lane_id]
    lane = lane[lane["y_known"].notna()]
    fitted, _, _, _ = forecast_sarimax(lane, max_horizon, config["arima_order"])
    for position, driver in enumerate(SARIMAX_DRIVERS):
        coefficient_rows.append(
            {
                "lane_id": lane_id,
                "driver": driver,
                "coefficient": fitted.params[position],
                "p_value": fitted.pvalues[position],
            }
        )
pd.DataFrame(coefficient_rows).round(4).to_csv(
    reports_folder / "sarimax_coefficients.csv", index=False
)

# ---------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------
main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
view = leaderboard[
    (leaderboard["lane_id"] == main_lane) & (leaderboard["period"] == "all")
]
print(f"MASE on {main_lane}, all periods:")
print(view.pivot(index="model", columns="horizon", values="mase").round(2).to_string())
view = interval_metrics[
    (interval_metrics["lane_id"] == main_lane) & (interval_metrics["period"] == "all")
]
print(f"80% interval coverage on {main_lane}, all periods:")
print(
    view.pivot(index="model", columns="horizon", values="coverage_80")
    .round(2)
    .to_string()
)

# ---------------------------------------------------------------
# 7. Public table for the dashboard, and MLflow
# ---------------------------------------------------------------
# Every forecast of naive, ARIMA, SARIMAX and LightGBM, with their 80%
# intervals, as percent changes from the rate known at the origin.
public = percent_change_table(all_point, forecasts)
public.to_csv(reports_folder / "backtest_forecasts.csv", index=False)
print(f"Saved reports/backtest_forecasts.csv ({len(public)} forecasts)")

# One parent run "driver_models", one child run per model.
child_settings = {
    "sarimax": {
        "order": config["arima_order"],
        "drivers": SARIMAX_DRIVERS,
        "interval": "model's own 80% interval",
    },
    "lightgbm": config["lightgbm"]
    | {"quantiles": quantile_levels, "n_features": len(LIGHTGBM_FEATURES)},
    "lightgbm_widened": {
        "interval": "conformal: widened by the 80% quantile of past misses",
        "min_past_scores": MIN_PAST_SCORES,
    },
    "naive_empirical": {"interval": "10% and 90% quantiles of past naive errors"},
}
artifact_paths = []
for file_name in [
    "leaderboard.csv",
    "interval_metrics.csv",
    "feature_importance.csv",
    "sarimax_coefficients.csv",
]:
    artifact_paths.append(reports_folder / file_name)
log_backtest_run(
    "driver_models",
    config,
    child_settings,
    leaderboard,
    artifact_paths,
    intervals=interval_metrics,
)
