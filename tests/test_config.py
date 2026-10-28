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


def test_exactly_one_main_lane():
    config = load_config()
    main_lanes = []
    for lane in config["lanes"]:
        if lane["role"] == "main":
            main_lanes.append(lane["id"])
    assert main_lanes == ["asia_middle_east"]


def test_every_lane_has_an_ncfi_route():
    config = load_config()
    for lane in config["lanes"]:
        assert lane["ncfi_route"] != ""


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
