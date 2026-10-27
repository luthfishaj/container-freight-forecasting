# ---------------------------------------------------------------
# Test: test_config.py
# Purpose: Checks that config/config.yaml loads and that its main
#          settings are consistent with the project scope.
# ---------------------------------------------------------------

import pytest

from hff.config import load_config


def test_config_loads_as_dictionary():
    config = load_config()
    assert isinstance(config, dict)


def test_lane_ids_are_unique():
    config = load_config()
    lane_ids = []
    for lane in config["lanes"]:
        lane_ids.append(lane["id"])
    assert len(lane_ids) == len(set(lane_ids))


def test_horizons_are_positive_and_sorted():
    config = load_config()
    horizons = config["horizons"]
    for horizon in horizons:
        assert horizon > 0
    assert horizons == sorted(horizons)


def test_quantiles_are_between_zero_and_one():
    config = load_config()
    for quantile in config["quantiles"]:
        assert 0 < quantile < 1


def test_missing_config_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing_path)
