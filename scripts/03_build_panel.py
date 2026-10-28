# ---------------------------------------------------------------
# Script: 03_build_panel.py
# Purpose: Builds the weekly panel used by every model: one row per
#          lane l and Friday t, with the target, its past values, the
#          drivers as known on that Friday, event flags and calendar
#          features.
# Input:   data/interim/*.csv (Phase 2), sql/*.sql, config/config.yaml
# Output:  warehouse/hff.duckdb  staging views, weekly driver tables
#                                and the panel table
#          data/processed/panel.parquet
#
# Columns of the panel:
#   rate, y            NCFI value and its log, y[l, t] (empty when no
#                      value was ever published for that week)
#   y_known            log rate as known on Friday t (see hff.features)
#   y_lag_k            y_known[l, t-k]
#   change_1w, change_mean_*w, change_std_*w
#                      weekly change of y_known and its rolling mean and
#                      standard deviation (momentum and volatility)
#   brent_usd, brent_change_4w, gprd, gulf_container_calls,
#   ningbo_container_calls
#                      drivers x[k, t - lag_k], already shifted by their
#                      publication lag
#   *_started          1 once a shock has started (known in real time)
#   period             calm / covid / red_sea / hormuz, for evaluation only
#   week_of_year, before_chinese_new_year, after_chinese_new_year
# ---------------------------------------------------------------

import duckdb
import numpy as np
import pandas as pd

from hff.config import PROJECT_ROOT, load_config
from hff.features import (
    add_calendar_features,
    add_evaluation_period,
    add_event_flags,
    add_known_log_rate,
    add_rolling_change_features,
    add_target_lags,
    shift_drivers_by_publication_lag,
)

config = load_config()
interim_folder = PROJECT_ROOT / config["paths"]["interim_data"]
sql_folder = PROJECT_ROOT / config["paths"]["sql"]
warehouse_path = PROJECT_ROOT / config["paths"]["warehouse"]
panel_path = PROJECT_ROOT / config["paths"]["panel"]
warehouse_path.parent.mkdir(parents=True, exist_ok=True)
panel_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 1. Warehouse: staging views and weekly driver tables (SQL)
# ---------------------------------------------------------------
print("Running the SQL transformations in the DuckDB warehouse...")
connection = duckdb.connect(str(warehouse_path))
for sql_file in sorted(sql_folder.glob("*.sql")):
    # DuckDB reads paths with forward slashes on every system.
    sql_text = sql_file.read_text(encoding="utf-8")
    sql_text = sql_text.replace("{interim}", interim_folder.as_posix())
    connection.execute(sql_text)
    print(f"  ran {sql_file.name}")

# ---------------------------------------------------------------
# 2. Lane rates: NCFI routes joined to the project lanes (SQL)
# ---------------------------------------------------------------
lanes = pd.DataFrame(config["lanes"])[["id", "ncfi_route", "role"]]
lanes = lanes.rename(columns={"id": "lane_id"})
connection.register("lanes", lanes)
lane_rates = connection.execute(
    """
    SELECT
        lanes.lane_id,
        week_ending_friday(stg_ncfi.week_date) AS week_date,
        stg_ncfi.rate,
        stg_ncfi.value_source
    FROM stg_ncfi
    JOIN lanes ON stg_ncfi.route = lanes.ncfi_route
    ORDER BY lanes.lane_id, week_date
    """
).df()
lane_rates["week_date"] = pd.to_datetime(lane_rates["week_date"])
print(f"  {len(lane_rates)} lane-week rates read")

# ---------------------------------------------------------------
# 3. Complete grid: every lane and every Friday
# ---------------------------------------------------------------
# Weeks without any published value stay in the panel with an empty
# rate, so that "one row = one week" always holds and lags are exact.
first_friday = pd.Timestamp(config["start_date"])
last_friday = lane_rates["week_date"].max()
all_fridays = pd.date_range(first_friday, last_friday, freq="W-FRI")

grid_parts = []
for lane in config["lanes"]:
    lane_grid = pd.DataFrame({"week_date": all_fridays})
    lane_grid["lane_id"] = lane["id"]
    lane_grid["role"] = lane["role"]
    grid_parts.append(lane_grid)
panel = pd.concat(grid_parts, ignore_index=True)
panel = panel.merge(lane_rates, on=["lane_id", "week_date"], how="left")
panel = panel.sort_values(["lane_id", "week_date"]).reset_index(drop=True)
panel["y"] = np.log(panel["rate"])

# ---------------------------------------------------------------
# 4. Target features (no look-ahead, see hff.features)
# ---------------------------------------------------------------
panel = add_known_log_rate(panel)
panel = add_target_lags(panel, config["target_lags"])
panel = add_rolling_change_features(panel, config["rolling_windows"])

# ---------------------------------------------------------------
# 5. Drivers, shifted by their publication lag
# ---------------------------------------------------------------
weekly_drivers = connection.execute("SELECT * FROM weekly_drivers").df()
weekly_drivers["week_date"] = pd.to_datetime(weekly_drivers["week_date"])
known_drivers = shift_drivers_by_publication_lag(
    weekly_drivers, config["driver_lag_weeks"]
)
# Change of the Brent price over four weeks, in log points. Both
# values are already lagged, so the change uses only published data.
log_brent = np.log(known_drivers["brent_usd"])
known_drivers["brent_change_4w"] = log_brent - log_brent.shift(4)
panel = panel.merge(known_drivers, on="week_date", how="left")

# ---------------------------------------------------------------
# 6. Events and calendar
# ---------------------------------------------------------------
panel = add_event_flags(panel, config["shock_periods"])
panel = add_evaluation_period(panel, config["shock_periods"])
panel = add_calendar_features(
    panel, config["chinese_new_year"], config["chinese_new_year_window_weeks"]
)

# ---------------------------------------------------------------
# 7. Save to the warehouse and to Parquet
# ---------------------------------------------------------------
panel = panel.sort_values(["lane_id", "week_date"]).reset_index(drop=True)
connection.register("panel_frame", panel)
connection.execute("CREATE OR REPLACE TABLE panel AS SELECT * FROM panel_frame")
connection.execute(f"COPY panel TO '{panel_path.as_posix()}' (FORMAT parquet)")
connection.close()

# ---------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------
print(
    f"Panel: {len(panel)} rows, {panel['lane_id'].nunique()} lanes, "
    f"{panel['week_date'].min().date()} to {panel['week_date'].max().date()}"
)
for lane_id in panel["lane_id"].unique():
    lane_rows = panel[panel["lane_id"] == lane_id]
    missing_weeks = int(lane_rows["rate"].isna().sum())
    print(f"  {lane_id}: {len(lane_rows)} weeks, {missing_weeks} without a value")
print("Weeks per evaluation period (per lane):")
print(
    panel[panel["lane_id"] == "asia_middle_east"]["period"].value_counts().to_string()
)
print(f"Saved {panel_path.relative_to(PROJECT_ROOT).as_posix()}")
