# ---------------------------------------------------------------
# Test: test_metrics.py
# Purpose: Checks each evaluation metric on small SYNTHETIC examples
#          whose results can be computed by hand.
# ---------------------------------------------------------------

import math

import pytest

from hff.metrics import (
    directional_accuracy,
    interval_coverage,
    majority_direction_share,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    pinball_loss,
)


def test_mean_absolute_error():
    # Errors 1, -2, 3 -> mean of 1, 2, 3 = 2
    assert mean_absolute_error([10, 20, 30], [9, 22, 27]) == pytest.approx(2.0)


def test_mean_absolute_percentage_error():
    # 10% and 20% errors -> 15%
    assert mean_absolute_percentage_error([100, 200], [110, 160]) == pytest.approx(15.0)


def test_mean_absolute_scaled_error():
    # |errors| 1 and 3, scales 2 and 3 -> mean of 0.5 and 1.0 = 0.75
    result = mean_absolute_scaled_error([5, 5], [4, 8], [2, 3])
    assert result == pytest.approx(0.75)


def test_directional_accuracy_ignores_no_change_forecasts():
    last_known = [10, 10, 10, 10]
    actual = [12, 8, 11, 9]
    # Forecasts: up (right), up (wrong), no change (ignored), down (right)
    forecast = [11, 11, 10, 9.5]
    assert directional_accuracy(last_known, actual, forecast) == pytest.approx(2 / 3)


def test_directional_accuracy_is_undefined_for_the_naive_forecast():
    result = directional_accuracy([10, 10], [12, 8], [10, 10])
    assert math.isnan(result)


def test_majority_direction_share():
    # Three moves up, one down -> 0.75
    assert majority_direction_share([0, 0, 0, 0], [1, 2, 3, -1]) == pytest.approx(0.75)


def test_pinball_loss_matches_the_worked_example():
    # Example from docs/concepts.md: alpha 0.9, quantile 1000.
    assert pinball_loss([1100], [1000], 0.9) == pytest.approx(90.0)
    assert pinball_loss([900], [1000], 0.9) == pytest.approx(10.0)


def test_interval_coverage_counts_bounds_as_inside():
    actual = [1, 5, 10, 11]
    lower = [0, 5, 0, 0]
    upper = [2, 6, 10, 10]
    assert interval_coverage(actual, lower, upper) == pytest.approx(0.75)
