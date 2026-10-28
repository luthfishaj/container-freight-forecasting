# ---------------------------------------------------------------
# Page: Monitoring
# Purpose: The weekly health checks written by scripts/12_monitor.py:
#          is every source up to date, is a driver far outside its
#          past-year range, how have past weekly forecasts done?
# Input:   reports/monitoring.csv, track_record.csv (once available),
#          reports/forecasts/
#
# Drift: z = (latest weekly value - mean of the previous 52 weeks) /
# standard deviation of those weeks; an alert above 3 in absolute value.
# ---------------------------------------------------------------

import pandas as pd
import streamlit as st
from common import REPORTS, data_footer, load_report, require_reports

CHECK_TITLES = {
    "freshness": "Data freshness (days since the last value)",
    "drift": "Driver drift (z-score against the past year)",
    "accuracy": "Live accuracy",
}

require_reports(["monitoring.csv"])
monitoring = load_report("monitoring.csv")
run_date = monitoring["run_date"].iloc[0]
alerts = int((monitoring["status"] == "ALERT").sum())

st.title("Monitoring")
if alerts == 0:
    st.success(f"OK: no alert in the checks of {run_date}.")
else:
    st.error(f"ALERT: {alerts} check(s) need attention ({run_date}).")

for check, title in CHECK_TITLES.items():
    rows = monitoring[monitoring["check"] == check]
    if rows.empty:
        continue
    st.subheader(title)
    table = rows[["item", "value", "threshold", "status", "detail"]].rename(
        columns={
            "item": "Item",
            "value": "Value",
            "threshold": "Threshold",
            "status": "Status",
            "detail": "Detail",
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

st.subheader("Weekly snapshots")
snapshot_files = sorted((REPORTS / "forecasts").glob("forecast_*.csv"), reverse=True)
snapshot_rows = []
for snapshot_file in snapshot_files:
    snapshot = pd.read_csv(snapshot_file)
    snapshot_rows.append(
        {
            "Week": snapshot["origin_week"].iloc[0],
            "Model": snapshot["model"].iloc[0],
            "Forecasts": len(snapshot),
        }
    )
st.dataframe(pd.DataFrame(snapshot_rows), hide_index=True, use_container_width=True)
st.caption(
    "Run `make weekly` after each Friday publication: it downloads the new data, "
    "adds a snapshot and reruns these checks. Snapshots are never changed, so "
    "they build a live track record."
)
data_footer()
