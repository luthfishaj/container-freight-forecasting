# ---------------------------------------------------------------
# Module: hff/metrics.py
# Purpose: Forecast evaluation metrics used in every leaderboard.
#
# Notation, for n forecasts i = 1..n:
#   y_i          actual value (log rate unless stated otherwise)
#   y_hat_i      forecast of y_i
#   y_last_i     last value known at the forecast origin t
#   e_i          forecast error, y_i - y_hat_i
#   scale_i      typical one-week move of the series before origin t:
#                mean of |y[t'] - y[t'-1]| over the training weeks
#
# Formulas:
#   MAE   = (1/n) * sum |e_i|
#   MAPE  = (100/n) * sum |e_i| / |y_i|        (on rates, not logs)
#   MASE  = (1/n) * sum |e_i| / scale_i
#           below the naive forecast's MASE = better than "no change"
#   Directional accuracy = share of forecasts whose predicted move
#           (y_hat_i - y_last_i) has the same sign as the actual move
#           (y_i - y_last_i); forecasts of exactly no change have no
#           direction and are left out
#   Pinball loss at level alpha, for a quantile forecast q_i:
#           alpha * (y_i - q_i)         if y_i >= q_i
#           (1 - alpha) * (q_i - y_i)   if y_i <  q_i
#           averaged over i; lower is better
#   Coverage = share of actual values inside [lower_i, upper_i]
# ---------------------------------------------------------------

import numpy as np


def mean_absolute_error(actual, forecast) -> float:
    """Average size of the forecast errors."""
    errors = np.asarray(actual) - np.asarray(forecast)
    return float(np.mean(np.abs(errors)))


def mean_absolute_percentage_error(actual, forecast) -> float:
    """Average error in percent of the actual value (use on rates, not logs)."""
    actual = np.asarray(actual)
    errors = actual - np.asarray(forecast)
    return float(100 * np.mean(np.abs(errors) / np.abs(actual)))


def mean_absolute_scaled_error(actual, forecast, scale) -> float:
    """Average error divided by the typical one-week move before each origin."""
    errors = np.asarray(actual) - np.asarray(forecast)
    return float(np.mean(np.abs(errors) / np.asarray(scale)))


def directional_accuracy(last_known, actual, forecast) -> float:
    """Share of forecasts that predicted the direction of the move correctly."""
    predicted_move = np.asarray(forecast) - np.asarray(last_known)
    actual_move = np.asarray(actual) - np.asarray(last_known)
    has_direction = predicted_move != 0
    if not has_direction.any():
        return float("nan")
    same_sign = np.sign(predicted_move) == np.sign(actual_move)
    return float(np.mean(same_sign[has_direction]))


def majority_direction_share(last_known, actual) -> float:
    """Score of the rule that always predicts the most frequent direction."""
    actual_move = np.asarray(actual) - np.asarray(last_known)
    share_up = np.mean(actual_move > 0)
    share_down = np.mean(actual_move < 0)
    return float(max(share_up, share_down))


def pinball_loss(actual, quantile_forecast, alpha: float) -> float:
    """Average pinball loss of a quantile forecast at level alpha."""
    gap = np.asarray(actual) - np.asarray(quantile_forecast)
    loss = np.where(gap >= 0, alpha * gap, (alpha - 1) * gap)
    return float(np.mean(loss))


def interval_coverage(actual, lower, upper) -> float:
    """Share of actual values that fall inside their prediction interval."""
    actual = np.asarray(actual)
    inside = (actual >= np.asarray(lower)) & (actual <= np.asarray(upper))
    return float(np.mean(inside))
