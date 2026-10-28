# ---------------------------------------------------------------
# Test: test_backtest.py
# Purpose: Checks the interval widening (conformal correction) and the
#          public percent-change table on small hand-made examples.
#
# All forecasts below are SYNTHETIC: invented values on the log scale.
# ---------------------------------------------------------------

import math

import pandas as pd

from hff.backtest import percent_change_table, widen_intervals


def synthetic_forecasts(n_weeks: int) -> pd.DataFrame:
    """One lane, horizon 1, an interval of +/-0.1 around a median of 0,
    and actual values 0.2 above the median (always outside)."""
    rows = []
    for week in range(n_weeks):
        origin = pd.Timestamp("2024-01-05") + pd.Timedelta(weeks=week)
        rows.append(
            {
                "model": "lightgbm",
                "lane_id": "lane_a",
                "horizon": 1,
                "origin_date": origin,
                "target_date": origin + pd.Timedelta(weeks=1),
                "y_last": 0.0,
                "y_forecast": 0.0,
                "y_actual": 0.2,
                "q10": -0.1,
                "q50": 0.0,
                "q90": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_widening_waits_for_enough_past_scores():
    forecasts = synthetic_forecasts(12)
    widened = widen_intervals(forecasts, "lightgbm_widened", min_past_scores=10)
    # Origin k (counted from 0) has k - 1 earlier forecasts whose target
    # week came before it, so only the last origin (k = 11) has 10.
    assert len(widened) == 1
    assert set(widened["model"]) == {"lightgbm_widened"}


def test_widening_adds_the_past_miss_to_both_ends():
    forecasts = synthetic_forecasts(15)
    widened = widen_intervals(forecasts, "lightgbm_widened", min_past_scores=10)
    # Every past score is 0.2 - 0.1 = 0.1, so the interval grows by 0.1
    # on each side and the median stays where it was.
    for row in widened.itertuples():
        assert math.isclose(row.q10, -0.2)
        assert math.isclose(row.q90, 0.2)
        assert row.q50 == 0.0


def test_percent_change_table_converts_logs_to_percent():
    point = synthetic_forecasts(1).drop(columns=["q10", "q50", "q90"])
    point["model"] = "arima"
    point["period"] = "calm"
    point["y_forecast"] = math.log(1.10)
    point["y_actual"] = math.log(0.80)
    no_intervals = synthetic_forecasts(1).iloc[0:0]
    table = percent_change_table(point, no_intervals)
    row = table.iloc[0]
    assert math.isclose(row["forecast_change_pct"], 10.0)
    assert math.isclose(row["actual_change_pct"], -20.0)
    assert math.isnan(row["low_change_pct"])  # ARIMA has no interval
