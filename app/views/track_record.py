# ---------------------------------------------------------------
# Page: Forecasts vs actual
# Purpose: Every backtest forecast of one model next to what actually
#          happened, week by week, with its 80% range; and the live
#          track record of the weekly snapshots once they mature.
# Input:   reports/backtest_forecasts.csv, leaderboard.csv,
#          interval_metrics.csv, track_record.csv (once available)
#
# Each point is the change from the origin week t to the target week
# t+h, in percent: the forecast change and the actual change.
# ---------------------------------------------------------------

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    REPORTS,
    data_footer,
    lane_picker,
    load_report,
    require_reports,
    shock_bands,
    show_chart,
)

from hff.charts import MODEL_COLORS, MODEL_LABELS, TEXT_PRIMARY

require_reports(["backtest_forecasts.csv", "leaderboard.csv", "interval_metrics.csv"])
forecasts = load_report("backtest_forecasts.csv")
forecasts["target_date"] = pd.to_datetime(forecasts["target_date"])

st.title("Forecasts vs actual")
columns = st.columns(3)
with columns[0]:
    lane_id = lane_picker()
with columns[1]:
    model = st.selectbox(
        "Model", list(MODEL_LABELS), index=2, format_func=MODEL_LABELS.get
    )
with columns[2]:
    horizon = st.selectbox("Weeks ahead", [1, 2, 4, 8, 12], index=0)

view = forecasts[
    (forecasts["lane_id"] == lane_id)
    & (forecasts["model"] == model)
    & (forecasts["horizon"] == horizon)
].sort_values("target_date")
view["inside"] = (view["actual_change_pct"] >= view["low_change_pct"]) & (
    view["actual_change_pct"] <= view["high_change_pct"]
)

# ---------------------------------------------------------------
# 1. Scores for this selection (whole backtest)
# ---------------------------------------------------------------
leaderboard = load_report("leaderboard.csv")
scores = leaderboard[
    (leaderboard["lane_id"] == lane_id)
    & (leaderboard["horizon"] == horizon)
    & (leaderboard["period"] == "all")
].set_index("model")
intervals = load_report("interval_metrics.csv")
interval_model = {"naive": "naive_empirical", "lightgbm": "lightgbm_widened"}.get(
    model, model
)
coverage = intervals[
    (intervals["lane_id"] == lane_id)
    & (intervals["horizon"] == horizon)
    & (intervals["period"] == "all")
    & (intervals["model"] == interval_model)
]["coverage_80"]

metric_columns = st.columns(3)
metric_columns[0].metric(
    'Error vs "no change"',
    f"{100 * (scores.loc[model, 'mase'] / scores.loc['naive', 'mase'] - 1):+.0f}%",
    help="MASE of the model divided by MASE of the naive forecast, minus 1.",
)
metric_columns[1].metric(
    "Mean error, % of the rate", f"{scores.loc[model, 'mape_pct']:.1f}%"
)
if len(coverage) > 0:
    metric_columns[2].metric("80% range held", f"{100 * coverage.iloc[0]:.0f}%")
else:
    metric_columns[2].metric("80% range held", "no range")

# ---------------------------------------------------------------
# 2. Forecast and actual change, week by week
# ---------------------------------------------------------------
color = MODEL_COLORS[model]
tooltip = [
    alt.Tooltip("target_date:T", title="Target week", format="%d %b %Y"),
    alt.Tooltip("origin_date:N", title="Forecast made in week"),
    alt.Tooltip("actual_change_pct:Q", title="Actual change, %", format="+.1f"),
    alt.Tooltip("forecast_change_pct:Q", title="Forecast change, %", format="+.1f"),
    alt.Tooltip("low_change_pct:Q", title="Range from, %", format="+.1f"),
    alt.Tooltip("high_change_pct:Q", title="Range to, %", format="+.1f"),
]
base = alt.Chart(view).encode(x=alt.X("target_date:T", title=None))
band = base.mark_area(color=color, opacity=0.18).encode(
    y="low_change_pct:Q", y2="high_change_pct:Q"
)
actual = base.mark_line(color=TEXT_PRIMARY, strokeWidth=1.2).encode(
    y=alt.Y("actual_change_pct:Q", title=f"Change over {horizon} weeks, %"),
    tooltip=tooltip,
)
forecast = base.mark_line(color=color, strokeWidth=2).encode(
    y="forecast_change_pct:Q", tooltip=tooltip
)
bands = shock_bands(view["target_date"].min(), view["target_date"].max())
st.subheader(f"{MODEL_LABELS[model]}, {horizon} week(s) ahead")
show_chart(bands + band + actual + forecast, 400)
st.caption(
    f"Black: actual change. Coloured line: {MODEL_LABELS[model]} forecast. "
    "Shaded: its 80% range (ARIMA has none). Grey bands: shock periods. "
    "Hover for the values of any week."
)

# ---------------------------------------------------------------
# 3. Live track record of the weekly snapshots
# ---------------------------------------------------------------
st.subheader("Live track record")
track_record_path = REPORTS / "track_record.csv"
snapshot_files = sorted((REPORTS / "forecasts").glob("forecast_*.csv"))
if track_record_path.exists():
    st.dataframe(pd.read_csv(track_record_path), hide_index=True)
else:
    first_week = (
        snapshot_files[0].stem.replace("forecast_", "") if snapshot_files else "-"
    )
    st.info(
        f"{len(snapshot_files)} weekly snapshot(s) saved, the first from the week "
        f"of {first_week}. Each is scored automatically by `make weekly` once its "
        "target weeks are published; the first 1-week forecast matures one week "
        "after its origin."
    )
data_footer()
