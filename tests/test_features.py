# ---------------------------------------------------------------
# Test: test_features.py
# Purpose: Leakage tests for the weekly panel features: no feature for
#          week t may depend on a value published after Friday t.
#
# All data below is SYNTHETIC: small invented series whose values make
# the expected results easy to check by hand.
# ---------------------------------------------------------------

import pandas as pd

from hff.features import (
    add_calendar_features,
    add_evaluation_period,
    add_event_flags,
    add_known_log_rate,
    add_rolling_change_features,
    add_target_lags,
    shift_drivers_by_publication_lag,
)

FRIDAYS = pd.date_range("2024-01-05", periods=10, freq="W-FRI")


def make_synthetic_panel() -> pd.DataFrame:
    """Two lanes, ten Fridays; lane_a has y = 0..9, lane_b has y = 100..109."""
    rows = []
    for lane_id, offset in [("lane_a", 0.0), ("lane_b", 100.0)]:
        for week_index, friday in enumerate(FRIDAYS):
            rows.append(
                {
                    "lane_id": lane_id,
                    "week_date": friday,
                    "y": offset + week_index,
                    "value_source": "current column",
                }
            )
    return pd.DataFrame(rows)


def build_target_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = add_known_log_rate(panel)
    panel = add_target_lags(panel, [1, 2])
    panel = add_rolling_change_features(panel, [3])
    return panel


def test_value_from_next_sheet_is_not_known_in_its_own_week():
    panel = make_synthetic_panel()
    # Week 5 of lane_a was only printed on the following week's sheet.
    panel.loc[5, "value_source"] = "previous column of next sheet"
    panel = add_known_log_rate(panel)
    assert panel.loc[5, "y_known"] == panel.loc[4, "y"]
    assert panel.loc[6, "y_known"] == panel.loc[6, "y"]


def test_target_lags_use_past_weeks_of_the_same_lane():
    panel = build_target_features(make_synthetic_panel())
    lane_b = panel[panel["lane_id"] == "lane_b"].reset_index(drop=True)
    # The first week of lane_b has no past: it must not borrow lane_a.
    assert pd.isna(lane_b.loc[0, "y_lag_1"])
    assert lane_b.loc[6, "y_lag_2"] == lane_b.loc[4, "y_known"]


def test_features_do_not_change_when_a_future_value_changes():
    original = build_target_features(make_synthetic_panel())

    changed_input = make_synthetic_panel()
    changed_input.loc[8, "y"] = 50.0  # lane_a, week 8
    changed = build_target_features(changed_input)

    feature_columns = ["y_known", "y_lag_1", "y_lag_2", "change_1w", "change_mean_3w"]
    weeks_before = original["week_date"] < FRIDAYS[8]
    pd.testing.assert_frame_equal(
        original.loc[weeks_before, feature_columns],
        changed.loc[weeks_before, feature_columns],
    )


def test_drivers_are_shifted_by_their_publication_lag():
    weekly_drivers = pd.DataFrame(
        {"week_date": FRIDAYS[:6], "brent_usd": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    known = shift_drivers_by_publication_lag(weekly_drivers, {"brent_usd": 2})
    known = known.set_index("week_date")["brent_usd"]
    # On the third Friday, only the value of the first Friday is known.
    assert pd.isna(known[FRIDAYS[1]])
    assert known[FRIDAYS[2]] == 1.0
    # The last value becomes known two weeks after its own week.
    assert known[FRIDAYS[7]] == 6.0


def test_event_flags_start_on_the_start_date():
    panel = make_synthetic_panel()
    periods = {"hormuz": {"start": str(FRIDAYS[3].date()), "end": None}}
    panel = add_event_flags(panel, periods)
    assert panel.loc[2, "hormuz_started"] == 0
    assert panel.loc[3, "hormuz_started"] == 1


def test_evaluation_period_uses_start_and_end():
    panel = make_synthetic_panel()
    periods = {
        "covid": {"start": str(FRIDAYS[2].date()), "end": str(FRIDAYS[4].date())}
    }
    panel = add_evaluation_period(panel, periods)
    assert panel.loc[1, "period"] == "calm"
    assert panel.loc[3, "period"] == "covid"
    assert panel.loc[5, "period"] == "calm"


def test_chinese_new_year_flags():
    panel = make_synthetic_panel()
    holiday = str(FRIDAYS[5].date())
    panel = add_calendar_features(panel, [holiday], window_weeks=2)
    assert list(panel.loc[2:7, "before_chinese_new_year"]) == [0, 1, 1, 0, 0, 0]
    assert list(panel.loc[2:7, "after_chinese_new_year"]) == [0, 0, 0, 1, 1, 0]
