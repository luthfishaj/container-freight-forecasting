# ---------------------------------------------------------------
# Page: Latest forecast
# Purpose: The champion's forecast from the latest week (or an earlier
#          snapshot): median and 80% range for 1 to 12 weeks ahead,
#          drawn after the last 26 weeks of history.
# Input:   reports/forecasts/forecast_<week>.csv, rebased_rates.csv
#
# Both history and forecast are shown as an index equal to 100 in the
# forecast's origin week: a median of 105 means +5% from that week.
# ---------------------------------------------------------------

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    LANE_IDS,
    MAIN_LANE,
    REPORTS,
    data_footer,
    lane_picker,
    load_report,
    require_reports,
    show_chart,
)

from hff.charts import LANE_COLORS, LANE_LABELS, MODEL_LABELS

HISTORY_WEEKS = 26


def snapshot_label(path) -> str:
    """Week of a snapshot file, for example forecast_2026-10-23.csv."""
    return path.stem.replace("forecast_", "")


require_reports(["rebased_rates.csv"])
snapshot_files = sorted((REPORTS / "forecasts").glob("forecast_*.csv"), reverse=True)
if not snapshot_files:
    st.error("No forecast snapshot yet. Run `make forecast` first.")
    st.stop()

st.title("Latest forecast")
left, right = st.columns(2)
with left:
    lane_id = lane_picker()
with right:
    snapshot_file = st.selectbox(
        "Forecast made in the week of",
        snapshot_files,
        format_func=snapshot_label,
    )
snapshot = pd.read_csv(snapshot_file, parse_dates=["origin_week", "target_week"])
origin_week = snapshot["origin_week"].iloc[0]
champion = snapshot["model"].iloc[0]
color = LANE_COLORS[lane_id]

# ---------------------------------------------------------------
# 1. Fan chart: history, then median and 80% range
# ---------------------------------------------------------------
rates = load_report("rebased_rates.csv")
rates["week_date"] = pd.to_datetime(rates["week_date"])
lane_rates = rates[(rates["lane_id"] == lane_id) & (rates["week_date"] <= origin_week)]
origin_value = lane_rates["index_first_week_100"].iloc[-1]
history = lane_rates.tail(HISTORY_WEEKS).copy()
history["index"] = 100 * history["index_first_week_100"] / origin_value

forecast = snapshot[snapshot["lane_id"] == lane_id].copy()
start = pd.DataFrame(
    {
        "target_week": [origin_week],
        "horizon": [0],
        "median_change_pct": [0.0],
        "low_change_pct": [0.0],
        "high_change_pct": [0.0],
    }
)
forecast = pd.concat([start, forecast], ignore_index=True)
forecast["median"] = 100 + forecast["median_change_pct"]
forecast["low"] = 100 + forecast["low_change_pct"]
forecast["high"] = 100 + forecast["high_change_pct"]

y_title = f"Index, week of {origin_week:%d %b %Y} = 100"
history_line = (
    alt.Chart(history)
    .mark_line(color=color, strokeWidth=2)
    .encode(
        x=alt.X("week_date:T", title=None),
        y=alt.Y("index:Q", title=y_title),
        tooltip=[
            alt.Tooltip("week_date:T", title="Week ending", format="%d %b %Y"),
            alt.Tooltip("index:Q", title="Index", format=".0f"),
        ],
    )
)
band = (
    alt.Chart(forecast)
    .mark_area(color=color, opacity=0.18)
    .encode(x="target_week:T", y="low:Q", y2="high:Q")
)
median_line = (
    alt.Chart(forecast)
    .mark_line(color=color, strokeDash=[6, 4], strokeWidth=2, point=True)
    .encode(
        x="target_week:T",
        y="median:Q",
        tooltip=[
            alt.Tooltip("horizon:Q", title="Weeks ahead"),
            alt.Tooltip("target_week:T", title="Week ending", format="%d %b %Y"),
            alt.Tooltip("median_change_pct:Q", title="Median change, %", format="+.1f"),
            alt.Tooltip("low_change_pct:Q", title="80% range from, %", format="+.1f"),
            alt.Tooltip("high_change_pct:Q", title="80% range to, %", format="+.1f"),
        ],
    )
)
st.subheader(f"{LANE_LABELS[lane_id]}: last {HISTORY_WEEKS} weeks and the forecast")
show_chart(band + history_line + median_line, 380)
st.caption(
    "Solid line: published index. Dashed line: median forecast. Shaded area: "
    "80% range, the band the rate should end up in about 8 weeks out of 10."
)

# ---------------------------------------------------------------
# 2. Table: every lane and horizon
# ---------------------------------------------------------------
st.subheader("All lanes, change from the origin week")
table_rows = []
for table_lane in LANE_IDS:
    lane_rows = snapshot[snapshot["lane_id"] == table_lane]
    row = {"Lane": LANE_LABELS[table_lane]}
    for forecast_row in lane_rows.itertuples():
        low = forecast_row.low_change_pct
        high = forecast_row.high_change_pct
        row[f"{forecast_row.horizon} wk"] = (
            f"{forecast_row.median_change_pct:+.0f}% ({low:+.0f}% to {high:+.0f}%)"
        )
    table_rows.append(row)
st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
st.markdown(
    f"Median and 80% range. Model: **{MODEL_LABELS.get(champion, champion)}**, "
    f"the champion: lowest error on the {LANE_LABELS[MAIN_LANE]} lane 4 weeks "
    "ahead over the last 52 weeks of the backtest. The range comes from the "
    'champion\'s own past errors. Beyond 4 weeks no model beats "no change", so '
    'read the long-horizon medians as "no change" and use the ranges.'
)
data_footer()
