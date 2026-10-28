# ---------------------------------------------------------------
# Module: hff/backtest.py
# Purpose: Shared pieces of the rolling-origin backtest: loading the
#          panel, choosing forecast origins, and turning a table of
#          forecasts into a leaderboard.
#
# Rolling-origin backtest (expanding window): for every Friday t from
# the first origin onward, a model is fitted on the weeks up to t only
# and forecasts y[l, t+h] for each horizon h. The forecast is then
# compared with the value actually published for week t+h. Every
# forecast therefore uses only information available at its origin.
#
# A forecast table has one row per (model, lane, origin, horizon):
#   y_last      y_known at the origin (last value known on Friday t)
#   y_forecast  forecast of the log rate y_hat[l, t+h | t]
#   y_actual    log rate published for week t+h
#   scale       mean |weekly change| of y_known up to t (for MASE)
#   period      evaluation period of the target week t+h
# ---------------------------------------------------------------

import numpy as np
import pandas as pd

from hff.metrics import (
    directional_accuracy,
    interval_coverage,
    majority_direction_share,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    pinball_loss,
)


def first_origin_position(y_known: pd.Series, warm_up_weeks: int) -> int:
    """Returns the row of the first origin, warm_up_weeks after the first value."""
    first_known = int(np.argmax(y_known.notna().to_numpy()))
    return first_known + warm_up_weeks


def training_scale(y_history: np.ndarray) -> float:
    """Mean absolute one-week change of the history (the MASE denominator)."""
    weekly_changes = np.abs(np.diff(y_history))
    return float(np.mean(weekly_changes))


def summarise_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Computes the metrics per model, lane, horizon and period (plus 'all')."""
    # Every forecast counts once in its own period and once in "all".
    all_periods = forecasts.copy()
    all_periods["period"] = "all"
    combined = pd.concat([forecasts, all_periods], ignore_index=True)

    rows = []
    group_columns = ["model", "lane_id", "horizon", "period"]
    for group_key, group in combined.groupby(group_columns):
        # MAE and MAPE are computed on index points, MASE on logs.
        actual_rate = np.exp(group["y_actual"])
        forecast_rate = np.exp(group["y_forecast"])
        row = dict(zip(group_columns, group_key))
        row["n_forecasts"] = len(group)
        row["mae_points"] = mean_absolute_error(actual_rate, forecast_rate)
        row["mape_pct"] = mean_absolute_percentage_error(actual_rate, forecast_rate)
        row["mase"] = mean_absolute_scaled_error(
            group["y_actual"], group["y_forecast"], group["scale"]
        )
        row["directional_accuracy"] = directional_accuracy(
            group["y_last"], group["y_actual"], group["y_forecast"]
        )
        row["majority_direction_share"] = majority_direction_share(
            group["y_last"], group["y_actual"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_intervals(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Coverage, width and pinball loss of the 10/50/90% quantile forecasts.

    Needs the columns q10, q50 and q90 (log scale). The width of the
    80% interval, q90 - q10, is reported in log points times 100, which
    is close to a width in percent of the rate.
    """
    all_periods = forecasts.copy()
    all_periods["period"] = "all"
    combined = pd.concat([forecasts, all_periods], ignore_index=True)

    rows = []
    group_columns = ["model", "lane_id", "horizon", "period"]
    for group_key, group in combined.groupby(group_columns):
        row = dict(zip(group_columns, group_key))
        row["n_forecasts"] = len(group)
        row["coverage_80"] = interval_coverage(
            group["y_actual"], group["q10"], group["q90"]
        )
        row["mean_width_pct"] = float(100 * (group["q90"] - group["q10"]).mean())
        losses = [
            pinball_loss(group["y_actual"], group["q10"], 0.1),
            pinball_loss(group["y_actual"], group["q50"], 0.5),
            pinball_loss(group["y_actual"], group["q90"], 0.9),
        ]
        row["pinball_loss"] = float(np.mean(losses))
        rows.append(row)
    return pd.DataFrame(rows)


def widen_intervals(
    model_forecasts: pd.DataFrame, new_model_name: str, min_past_scores: int
) -> pd.DataFrame:
    """Widens 10-90% intervals from the model's own past misses.

    Conformalized quantile regression (Romano, Patterson and Candes,
    2019), per lane and horizon:
        score_i  = max(q10_i - y_i, y_i - q90_i)   distance of the actual
                   value outside the interval (negative when inside)
        Q_t      = 80% quantile of the scores of the forecasts whose
                   target week was published before the origin t
        interval = [q10 - Q_t, q90 + Q_t]
    The median q50 is unchanged. Origins with fewer than min_past_scores
    earlier scores get no widened interval. Returns new rows named
    new_model_name.
    """
    widened_rows = []
    for _, group in model_forecasts.groupby(["lane_id", "horizon"]):
        below = group["q10"] - group["y_actual"]
        above = group["y_actual"] - group["q90"]
        scores = np.maximum(below, above)
        for index in group.index:
            published = group["target_date"] < group.loc[index, "origin_date"]
            past_scores = scores[published]
            if len(past_scores) < min_past_scores:
                continue
            correction = past_scores.quantile(0.8)
            row = group.loc[index].to_dict()
            row["model"] = new_model_name
            # A negative correction narrows the interval; q50 stays inside.
            row["q10"] = min(row["q10"] - correction, row["q50"])
            row["q90"] = max(row["q90"] + correction, row["q50"])
            widened_rows.append(row)
    return pd.DataFrame(widened_rows)


# Model whose rows give each point model its 80% interval in the
# public table: naive with the naive-error benchmark, LightGBM with its
# widened interval, SARIMAX with its own. ARIMA has none.
INTERVAL_SOURCE = {
    "naive": "naive_empirical",
    "lightgbm": "lightgbm_widened",
    "sarimax": "sarimax",
}
PUBLIC_MODELS = ["naive", "arima", "sarimax", "lightgbm"]


def percent_change_table(
    point_forecasts: pd.DataFrame, interval_forecasts: pd.DataFrame
) -> pd.DataFrame:
    """Every forecast of the public models as percent changes.

    Each value is written as a change from the rate known at the
    origin, so that no index value appears in the public table:
        change_pct = 100 * (exp(value - y_last) - 1)
    point_forecasts holds y_forecast for every model; interval_forecasts
    holds the q10 and q90 columns of the interval models.
    """
    keys = ["lane_id", "origin_date", "horizon"]
    intervals = interval_forecasts.copy()
    intervals["origin_date"] = pd.to_datetime(intervals["origin_date"])
    in_public = point_forecasts["model"].isin(PUBLIC_MODELS)
    parts = []
    for model_name, group in point_forecasts[in_public].groupby("model"):
        columns = keys + ["target_date", "period", "y_last", "y_forecast", "y_actual"]
        part = group[columns].copy()
        part["origin_date"] = pd.to_datetime(part["origin_date"])
        part["model"] = model_name
        if model_name in INTERVAL_SOURCE:
            source = intervals[intervals["model"] == INTERVAL_SOURCE[model_name]]
            part = part.merge(source[keys + ["q10", "q90"]], on=keys, how="left")
        else:
            part["q10"] = np.nan
            part["q90"] = np.nan
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)

    change_columns = {
        "forecast_change_pct": "y_forecast",
        "actual_change_pct": "y_actual",
        "low_change_pct": "q10",
        "high_change_pct": "q90",
    }
    for column, source_column in change_columns.items():
        table[column] = 100 * (np.exp(table[source_column] - table["y_last"]) - 1)
    table["origin_date"] = table["origin_date"].dt.date
    table["target_date"] = pd.to_datetime(table["target_date"]).dt.date
    first_columns = ["model", "lane_id", "horizon", "origin_date", "target_date"]
    return table[first_columns + ["period"] + list(change_columns)].round(2)
