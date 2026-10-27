# Container Freight Rate Forecasting

Weekly forecasts of container spot freight rates on four Asia trade lanes (North Europe, Mediterranean, Persian Gulf, US West Coast), 1 to 12 weeks ahead, with prediction intervals. The project also measures how the 2026 Strait of Hormuz disruption affected each lane and through which channel (fuel, rerouting, capacity), and illustrates how a pricing team could use the forecasts.

Only public data is used. The modelling period starts in January 2018.

**Status:** project setup. Data collection and modelling are the next steps.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and GNU `make`.

```
uv sync
```

This installs Python 3.12 and the exact package versions recorded in `uv.lock`. Secrets, when needed, go in a `.env` file created from `.env.example`.

## Commands

| Command | What it does |
|---|---|
| `make lint` | Checks code style and likely errors |
| `make format` | Applies the project code style |
| `make test` | Runs the unit tests |
| `make check` | Runs lint and tests, as in the CI |

## Project structure

```
config/config.yaml   settings: lanes, horizons, quantiles, dates, paths
data/                raw, interim and processed data (not tracked; rebuilt by the scripts)
sql/                 SQL transformations for the local DuckDB warehouse
scripts/             numbered pipeline scripts, run in order
src/hff/             small shared helpers
tests/               unit tests
reports/             figures, results report and weekly forecast snapshots
app/                 dashboard
```
