# ---------------------------------------------------------------
# Script: 12_monitor.py
# Purpose: Weekly health check of the forecasting pipeline, run after
#          11_weekly_forecast.py ("make weekly"). Three checks:
#            freshness  is every source up to date?
#            drift      is the latest value of each driver unusual
#                       compared with the past year?
#            accuracy   how did past weekly snapshots perform once the
#                       real rates were published (live track record)?
# Input:   data/interim/*.csv, data/processed/panel.parquet,
#          data/processed/forecast_snapshots/*.csv
# Output:  reports/monitoring.csv     one row per check, status OK/ALERT
#          reports/track_record.csv   live accuracy by horizon (once
#                                     snapshots have matured)
#          MLflow experiment "weekly_forecast", run "monitor <date>"
#
# Drift check: z = (latest weekly value - mean of the previous 52
# weeks) / standard deviation of those weeks. |z| > 3 means the driver
# is far outside its recent range, so the models are extrapolating.
# ---------------------------------------------------------------

from datetime import date

import mlflow
import numpy as np
import pandas as pd

from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel
from hff.tracking import log_finite_metrics, start_tracking

config = load_config()
interim_folder = PROJECT_ROOT / config["paths"]["interim_data"]
snapshot_folder = (
    PROJECT_ROOT / config["paths"]["processed_data"] / "forecast_snapshots"
)
reports_folder = PROJECT_ROOT / "reports"
today = pd.Timestamp(date.today())
checks = []

# ---------------------------------------------------------------
# 1. Freshness: days since the last value of each source
# ---------------------------------------------------------------
# Thresholds: the normal publication delay of each source plus a few
# days of tolerance (see the publication lags in config/config.yaml).
FRESHNESS_DAYS = {
    "ncfi_weekly.csv": ("week_date", 10),
    "brent.csv": ("date", 14),
    "gpr_daily.csv": ("date", 5),
    "portwatch_ports.csv": ("date", 18),
}
for file_name, (date_column, max_days) in FRESHNESS_DAYS.items():
    table = pd.read_csv(interim_folder / file_name, usecols=[date_column])
    last_date = pd.to_datetime(table[date_column]).max()
    age_days = int((today - last_date).days)
    checks.append(
        {
            "check": "freshness",
            "item": file_name,
            "value": age_days,
            "threshold": max_days,
            "status": "OK" if age_days <= max_days else "ALERT",
            "detail": f"last value {last_date.date()}",
        }
    )

# ---------------------------------------------------------------
# 2. Drift: latest known driver value vs the previous 52 weeks
# ---------------------------------------------------------------
panel = load_panel(config)
main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
drivers = panel[panel["lane_id"] == main_lane].set_index("week_date")
for driver in ["brent_usd", "gprd", "gulf_container_calls"]:
    series = drivers[driver].dropna()
    latest = series.iloc[-1]
    past_year = series.iloc[-53:-1]
    z_score = (latest - past_year.mean()) / past_year.std()
    checks.append(
        {
            "check": "drift",
            "item": driver,
            "value": round(float(z_score), 2),
            "threshold": 3,
            "status": "OK" if abs(z_score) <= 3 else "ALERT",
            "detail": f"latest {latest:.1f} vs past-year mean {past_year.mean():.1f}",
        }
    )

# ---------------------------------------------------------------
# 3. Live track record: snapshots whose target weeks are published
# ---------------------------------------------------------------
actual = panel[["lane_id", "week_date", "y"]].rename(
    columns={"week_date": "target_week"}
)
snapshot_files = sorted(snapshot_folder.glob("forecast_*.csv"))
matured = []
for snapshot_file in snapshot_files:
    snapshot = pd.read_csv(snapshot_file, parse_dates=["target_week"])
    snapshot = snapshot.merge(actual, on=["lane_id", "target_week"], how="inner")
    matured.append(snapshot.dropna(subset=["y"]))

if matured and sum(len(part) for part in matured) > 0:
    results = pd.concat(matured, ignore_index=True)
    # Error in percent of the actual rate: 100 * |exp(q50 - y) - 1|.
    results["error_pct"] = 100 * (np.exp(results["q50"] - results["y"]) - 1).abs()
    results["inside_interval"] = (results["y"] >= results["q10"]) & (
        results["y"] <= results["q90"]
    )
    track_record = (
        results.groupby(["lane_id", "horizon"])
        .agg(
            n_forecasts=("error_pct", "size"),
            mean_error_pct=("error_pct", "mean"),
            coverage_80=("inside_interval", "mean"),
        )
        .reset_index()
        .round(3)
    )
    track_record.to_csv(reports_folder / "track_record.csv", index=False)
    main_record = track_record[track_record["lane_id"] == main_lane]
    coverage = float(
        np.average(main_record["coverage_80"], weights=main_record["n_forecasts"])
    )
    n_forecasts = int(main_record["n_forecasts"].sum())
    # With fewer than 20 matured forecasts, coverage is too noisy to judge.
    status = "OK" if n_forecasts < 20 or 0.65 <= coverage <= 0.95 else "ALERT"
    checks.append(
        {
            "check": "accuracy",
            "item": f"{main_lane} interval coverage",
            "value": round(coverage, 2),
            "threshold": "0.65 to 0.95",
            "status": status,
            "detail": f"{n_forecasts} matured forecasts",
        }
    )
else:
    checks.append(
        {
            "check": "accuracy",
            "item": "live track record",
            "value": 0,
            "threshold": "-",
            "status": "OK",
            "detail": f"{len(snapshot_files)} snapshots, none matured yet",
        }
    )

# ---------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------
monitoring = pd.DataFrame(checks)
monitoring.insert(0, "run_date", today.date())
monitoring.to_csv(reports_folder / "monitoring.csv", index=False)
print(monitoring.drop(columns="run_date").to_string(index=False))
alerts = int((monitoring["status"] == "ALERT").sum())
print(f"Overall: {'ALERT' if alerts else 'OK'} ({alerts} alerts)")

# ---------------------------------------------------------------
# 5. MLflow: one run per check date
# ---------------------------------------------------------------
start_tracking("weekly_forecast")
with mlflow.start_run(run_name=f"monitor {today.date()}"):
    monitor_metrics = {"alerts": alerts}
    for row in monitoring.itertuples():
        monitor_metrics[f"{row.check}_{row.item}"] = float(row.value)
    log_finite_metrics(monitor_metrics)
    mlflow.log_artifact(str(reports_folder / "monitoring.csv"))
