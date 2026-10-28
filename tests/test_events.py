# ---------------------------------------------------------------
# Test: test_events.py
# Purpose: Checks the hand-built event calendar data/events.csv:
#          valid and sorted dates, known categories, and a source URL
#          on every row (no event enters the models without a source).
# ---------------------------------------------------------------

import pandas as pd

from hff.config import PROJECT_ROOT

EVENTS_PATH = PROJECT_ROOT / "data" / "events.csv"
KNOWN_CATEGORIES = ["pandemic", "canal_blockage", "red_sea", "hormuz"]


def load_events() -> pd.DataFrame:
    return pd.read_csv(EVENTS_PATH)


def test_events_have_the_expected_columns():
    events = load_events()
    assert list(events.columns) == ["date", "event", "category", "source_url"]


def test_event_dates_are_valid_and_sorted():
    events = load_events()
    dates = pd.to_datetime(events["date"], format="%Y-%m-%d")
    assert dates.is_monotonic_increasing


def test_event_categories_are_known():
    events = load_events()
    for category in events["category"]:
        assert category in KNOWN_CATEGORIES


def test_every_event_has_a_source_url():
    events = load_events()
    for url in events["source_url"]:
        assert url.startswith("https://")
