# ---------------------------------------------------------------
# Makefile: short names for the main project commands.
# "make all" rebuilds every result from the public sources, in order:
# data -> panel -> eda -> baselines -> models -> stress -> scenarios
# -> pricing. "make weekly" is the routine run once a week.
# ---------------------------------------------------------------

.PHONY: install lint format test check data rates drivers panel eda baselines models stress scenarios pricing forecast monitor weekly all dashboard mlflow

# Creates the virtual environment and installs the locked dependencies.
install:
	uv sync

# Reports style problems and likely errors without changing any file.
lint:
	uv run ruff check .
	uv run ruff format --check .

# Rewrites files to the project style and applies safe automatic fixes.
format:
	uv run ruff format .
	uv run ruff check --fix .

# Runs the unit tests.
test:
	uv run pytest

# Runs the same checks as the CI workflow.
check: lint test

# Downloads the NCFI data sheets and builds the weekly rate table.
# The first run takes several hours (30 seconds between requests to
# the news site); later runs only fetch the new weeks.
rates:
	uv run python scripts/01_ingest_rates.py
	uv run python scripts/01b_parse_rates.py

# Downloads the driver series and builds their cleaned tables.
drivers:
	uv run python scripts/02_ingest_drivers.py
	uv run python scripts/02b_clean_drivers.py

# Builds all interim data from the public sources.
data: rates drivers

# Builds the weekly panel (DuckDB warehouse and Parquet file).
panel:
	uv run python scripts/03_build_panel.py

# Exploratory figures and summary tables.
eda:
	uv run python scripts/04_eda.py

# Rolling-origin backtest of the baselines (about 5 minutes).
baselines:
	uv run python scripts/05_backtest_baselines.py

# Rolling-origin backtest of the driver models (about 8 minutes).
models:
	uv run python scripts/06_backtest_models.py

# Hormuz and Red Sea stress tests.
stress:
	uv run python scripts/07_shock_stress_test.py

# Conditional scenarios for the main lane.
scenarios:
	uv run python scripts/09_scenarios.py

# Lock-or-spot pricing illustration.
pricing:
	uv run python scripts/10_pricing_policy.py

# Champion forecast from the latest week, saved as a dated snapshot.
forecast:
	uv run python scripts/11_weekly_forecast.py

# Freshness, drift and live accuracy checks.
monitor:
	uv run python scripts/12_monitor.py

# Weekly routine: new data, panel, forecast snapshot, health checks.
# The backtests are not rerun every week; rerun "make all" monthly.
weekly: data panel forecast monitor

# Every step, from downloads to the pricing illustration.
all: data panel eda baselines models stress scenarios pricing

# Interactive dashboard of the results (http://localhost:8501). It
# reads only the public files in reports/, written by the steps above.
# Both tools are started with "python -m": on some Windows machines the
# launcher files (mlflow.exe, streamlit.exe) are blocked by Smart App
# Control, while python.exe is allowed.
dashboard:
	uv run python -m streamlit run app/dashboard.py

# MLflow interface: every backtest, stress test, scenario, pricing and
# weekly forecast run, with parameters, metrics and files
# (http://127.0.0.1:5000).
mlflow:
	uv run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
