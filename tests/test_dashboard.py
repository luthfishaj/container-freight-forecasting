# ---------------------------------------------------------------
# Test: test_dashboard.py
# Purpose: Runs every page of the Streamlit dashboard headless, on the
#          result files in reports/, and checks that none raises an
#          error. It does not check how the charts look.
#
# The test is skipped when the result files have not been built yet
# (run "make all" first).
# ---------------------------------------------------------------

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "app" / "dashboard.py"
REQUIRED = [
    "rebased_rates.csv",
    "eda_hormuz.csv",
    "leaderboard.csv",
    "interval_metrics.csv",
    "backtest_forecasts.csv",
    "stress_test_errors.csv",
    "scenario_grid.csv",
    "pricing_policy.csv",
    "monitoring.csv",
]
PAGES = [
    "views/overview.py",
    "views/forecast.py",
    "views/accuracy.py",
    "views/track_record.py",
    "views/stress_test.py",
    "views/scenarios.py",
    "views/pricing.py",
    "views/monitoring.py",
]


def reports_are_built() -> bool:
    for file_name in REQUIRED:
        if not (PROJECT_ROOT / "reports" / file_name).exists():
            return False
    return True


@pytest.mark.skipif(not reports_are_built(), reason="run make all first")
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_error(page):
    app = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    app.switch_page(page)
    app.run()
    assert len(app.exception) == 0, app.exception
    assert len(app.error) == 0, app.error
    assert len(app.title) == 1
