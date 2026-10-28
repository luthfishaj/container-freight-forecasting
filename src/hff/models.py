# ---------------------------------------------------------------
# Module: hff/models.py
# Purpose: The forecasting models and their features, shared by the
#          backtests (05, 06), the stress test (07), the scenarios (09)
#          and the weekly forecast (11), so every step uses exactly the
#          same model definitions.
#
# Notation (see config/config.yaml): y_known[l, t] is the log rate as
# known on Friday t; a forecast path holds y_hat[l, t+h | t] for
# h = 1..steps.
#
# Baselines (on one lane's history of y_known):
#   naive           y_hat = y[t]
#   seasonal_naive  y_hat = y[t+h-52]
#   drift           y_hat = y[t] + h * average weekly change of the history
#   ets             exponential smoothing, damped additive trend
#   arima           ARIMA(p, d, q) from the configuration
# Driver models:
#   sarimax         ARIMA with regressors x = log Brent, log GPR,
#                   log(1 + Gulf container port calls); future driver
#                   values held at their last known value unless a path
#                   is given (scenarios)
#   lightgbm        quantile gradient-boosted trees on all lanes, one
#                   model per horizon h and level alpha; target is the
#                   change y[l, t+h] - y_known[l, t]
# ---------------------------------------------------------------

import lightgbm
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

SARIMAX_DRIVERS = ["log_brent", "log_gprd", "log_gulf_calls"]


# ---------------------------------------------------------------
# Features
# ---------------------------------------------------------------
def driver_regressors(drivers: pd.DataFrame) -> pd.DataFrame:
    """Turns raw driver values into the log regressors used by the models."""
    # log(1 + x) for port calls, which can be zero.
    regressors = pd.DataFrame(index=drivers.index)
    regressors["log_brent"] = np.log(drivers["brent_usd"])
    regressors["log_gprd"] = np.log(drivers["gprd"])
    regressors["log_gulf_calls"] = np.log1p(drivers["gulf_container_calls"])
    return regressors


def lightgbm_features(config: dict) -> list[str]:
    """Names of the LightGBM features."""
    return (
        ["lane_code"]
        + [f"change_{lag}w" for lag in config["target_lags"]]
        + ["change_mean_4w", "change_std_4w", "change_mean_12w", "change_std_12w"]
        + ["log_brent", "brent_change_4w", "log_gprd"]
        + ["log_gulf_calls", "gulf_calls_change_4w", "log_ningbo_calls"]
        + ["covid_started", "red_sea_started", "hormuz_started"]
        + ["week_of_year", "before_chinese_new_year", "after_chinese_new_year"]
    )


def add_model_features(
    panel: pd.DataFrame, config: dict, horizons: list[int]
) -> pd.DataFrame:
    """Adds the model features and, for each horizon, the LightGBM target."""
    lane_ids = sorted(panel["lane_id"].unique())
    lane_codes = {}
    for code, lane_id in enumerate(lane_ids):
        lane_codes[lane_id] = code
    panel["lane_code"] = panel["lane_id"].map(lane_codes)

    # Past changes over k weeks, y_known[t] - y_known[t-k]: scale-free,
    # so one model can serve every lane.
    for lag in config["target_lags"]:
        panel[f"change_{lag}w"] = panel["y_known"] - panel[f"y_lag_{lag}"]

    regressors = driver_regressors(panel)
    for column in SARIMAX_DRIVERS:
        panel[column] = regressors[column]
    panel["log_ningbo_calls"] = np.log1p(panel["ningbo_container_calls"])
    previous_gulf_calls = panel.groupby("lane_id")["log_gulf_calls"].shift(4)
    panel["gulf_calls_change_4w"] = panel["log_gulf_calls"] - previous_gulf_calls

    # Row position t within each lane, used to pick origins.
    panel["t"] = panel.groupby("lane_id").cumcount()

    # LightGBM target for horizon h: change from the value known at t
    # to the value published for week t+h.
    for horizon in horizons:
        future_y = panel.groupby("lane_id")["y"].shift(-horizon)
        panel[f"target_{horizon}"] = future_y - panel["y_known"]
    return panel


# ---------------------------------------------------------------
# Baselines: each returns a forecast path for h = 1..steps
# ---------------------------------------------------------------
def forecast_naive(history: np.ndarray, steps: int) -> np.ndarray:
    return np.repeat(history[-1], steps)


def forecast_seasonal_naive(history: np.ndarray, steps: int) -> np.ndarray:
    # For horizon h, the value of week t+h-52, already known at t.
    path = []
    for horizon in range(1, steps + 1):
        path.append(history[len(history) - 52 + horizon - 1])
    return np.array(path)


def forecast_drift(history: np.ndarray, steps: int) -> np.ndarray:
    average_change = (history[-1] - history[0]) / (len(history) - 1)
    return history[-1] + np.arange(1, steps + 1) * average_change


def forecast_ets(history: np.ndarray, steps: int) -> np.ndarray:
    model = ExponentialSmoothing(history, trend="add", damped_trend=True)
    return model.fit().forecast(steps)


def forecast_arima(history: np.ndarray, steps: int, order: list[int]) -> np.ndarray:
    return ARIMA(history, order=tuple(order)).fit().forecast(steps)


# ---------------------------------------------------------------
# Driver models
# ---------------------------------------------------------------
def fit_sarimax(y_history: np.ndarray, regressors: np.ndarray, order: list[int]):
    """Fits the SARIMAX model on one lane's history."""
    model = SARIMAX(y_history, exog=regressors, order=tuple(order))
    return model.fit(disp=False)


def forecast_sarimax(history: pd.DataFrame, steps: int, order: list[int]) -> tuple:
    """Returns the fitted model and its mean, 10% and 90% forecast paths."""
    # A driver missing in some week (holiday) keeps its last known value.
    regressors = history[SARIMAX_DRIVERS].ffill().bfill().to_numpy()
    fitted = fit_sarimax(history["y_known"].to_numpy(), regressors, order)
    future_regressors = np.repeat(regressors[-1:], steps, axis=0)
    prediction = fitted.get_forecast(steps, exog=future_regressors)
    interval = prediction.conf_int(alpha=0.2)  # 80% interval
    return fitted, prediction.predicted_mean, interval[:, 0], interval[:, 1]


def fit_lightgbm(training_rows: pd.DataFrame, horizon: int, alpha: float, config: dict):
    """Fits one LightGBM quantile model for horizon h and level alpha."""
    model = lightgbm.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=config["random_seed"],
        **config["lightgbm"],
    )
    features = lightgbm_features(config)
    model.fit(training_rows[features], training_rows[f"target_{horizon}"])
    return model
