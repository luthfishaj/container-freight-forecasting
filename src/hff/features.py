# ---------------------------------------------------------------
# Module: hff/features.py
# Purpose: Builds the features of the weekly panel. Every function
#          follows one rule: a feature for week t uses only values
#          that were published by Friday t (no look-ahead).
#
# Notation (see config/config.yaml):
#   y[l, t]        log rate of lane l in week t, as finally known
#   y_known[l, t]  log rate as known on Friday t: the value printed on
#                  the sheet of week t, or else the last value known
#                  before (a value filled from the next week's sheet
#                  only becomes known one week later)
#   x[k, t]        weekly value of driver k
# The panel must be sorted by lane_id and week_date, one row per lane
# and Friday, with no missing Fridays.
# ---------------------------------------------------------------

import pandas as pd


def add_known_log_rate(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds y_known: the log rate as known on each Friday."""
    # Only values printed on the sheet of their own week count as
    # known that Friday; other weeks carry the last known value.
    published_that_week = panel["value_source"] == "current column"
    known_values = panel["y"].where(published_that_week)
    panel["y_known"] = known_values.groupby(panel["lane_id"]).ffill()
    return panel


def add_target_lags(panel: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Adds y_lag_k = y_known[l, t-k] for each k, lane by lane."""
    for lag in lags:
        panel[f"y_lag_{lag}"] = panel.groupby("lane_id")["y_known"].shift(lag)
    return panel


def add_rolling_change_features(
    panel: pd.DataFrame, windows: list[int]
) -> pd.DataFrame:
    """Adds the weekly change of y_known and its rolling mean and spread."""
    # change_1w[l, t] = y_known[l, t] - y_known[l, t-1], close to the
    # percentage change of the rate over the last week.
    previous_week = panel.groupby("lane_id")["y_known"].shift(1)
    panel["change_1w"] = panel["y_known"] - previous_week

    # The rolling windows end at week t, so they only contain changes
    # that are already known on Friday t.
    for window in windows:
        mean_column = f"change_mean_{window}w"
        std_column = f"change_std_{window}w"
        panel[mean_column] = float("nan")
        panel[std_column] = float("nan")
        for lane_id in panel["lane_id"].unique():
            lane_rows = panel["lane_id"] == lane_id
            lane_changes = panel.loc[lane_rows, "change_1w"]
            panel.loc[lane_rows, mean_column] = lane_changes.rolling(window).mean()
            panel.loc[lane_rows, std_column] = lane_changes.rolling(window).std()
    return panel


def shift_drivers_by_publication_lag(
    weekly_drivers: pd.DataFrame, lag_weeks: dict[str, int]
) -> pd.DataFrame:
    """Returns each driver as known on Friday t: x[k, t - lag_k]."""
    # A complete grid of Fridays is built first, so that shifting by
    # one row always means shifting by exactly one week.
    drivers = weekly_drivers.set_index("week_date").sort_index()
    all_fridays = pd.date_range(drivers.index.min(), drivers.index.max(), freq="W-FRI")
    drivers = drivers.reindex(all_fridays)

    # The grid is extended by the largest lag, so that the latest
    # published values reach the Fridays when they become known.
    largest_lag = max(lag_weeks.values())
    extra_fridays = pd.date_range(
        all_fridays[-1], periods=largest_lag + 1, freq="W-FRI"
    )[1:]
    drivers = drivers.reindex(all_fridays.append(extra_fridays))

    known_drivers = pd.DataFrame(index=drivers.index)
    for column, lag in lag_weeks.items():
        known_drivers[column] = drivers[column].shift(lag)
    known_drivers.index.name = "week_date"
    return known_drivers.reset_index()


def add_event_flags(panel: pd.DataFrame, shock_periods: dict) -> pd.DataFrame:
    """Adds one 0/1 flag per shock: 1 from the week the shock started."""
    # Only start dates are used: on Friday t it is known that a shock
    # has started, but not when it will end. End dates are used only
    # to label evaluation periods (add_evaluation_period).
    for name, period in shock_periods.items():
        start = pd.Timestamp(period["start"])
        panel[f"{name}_started"] = (panel["week_date"] >= start).astype(int)
    return panel


def add_evaluation_period(panel: pd.DataFrame, shock_periods: dict) -> pd.DataFrame:
    """Labels each week calm, covid, red_sea or hormuz (for evaluation only)."""
    panel["period"] = "calm"
    for name, period in shock_periods.items():
        in_period = panel["week_date"] >= pd.Timestamp(period["start"])
        if period["end"] is not None:
            in_period = in_period & (panel["week_date"] <= pd.Timestamp(period["end"]))
        panel.loc[in_period, "period"] = name
    return panel


def add_calendar_features(
    panel: pd.DataFrame, holiday_dates: list[str], window_weeks: int
) -> pd.DataFrame:
    """Adds the week of the year and Chinese New Year flags."""
    panel["week_of_year"] = panel["week_date"].dt.isocalendar().week.astype(int)

    # The holiday dates are fixed in advance by the lunar calendar, so
    # using them for future weeks is not look-ahead.
    window = pd.Timedelta(weeks=window_weeks)
    before_holiday = pd.Series(False, index=panel.index)
    after_holiday = pd.Series(False, index=panel.index)
    for holiday_text in holiday_dates:
        holiday = pd.Timestamp(holiday_text)
        before_holiday = before_holiday | (
            (panel["week_date"] >= holiday - window) & (panel["week_date"] < holiday)
        )
        after_holiday = after_holiday | (
            (panel["week_date"] >= holiday) & (panel["week_date"] < holiday + window)
        )
    panel["before_chinese_new_year"] = before_holiday.astype(int)
    panel["after_chinese_new_year"] = after_holiday.astype(int)
    return panel
