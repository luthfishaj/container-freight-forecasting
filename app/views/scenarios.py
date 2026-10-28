# ---------------------------------------------------------------
# Page: Scenario explorer
# Purpose: What the SARIMAX model of the Middle East lane implies IF
#          the drivers move to chosen levels: Brent, geopolitical risk
#          and Gulf container port calls. Conditional "what if"
#          results, not forecasts.
# Input:   reports/scenario_grid.csv (100 precomputed driver paths),
#          scenarios.csv, scenario_assumptions.csv
#
# Each driver moves in a straight line from its latest value to the
# chosen level over 8 weeks, then stays flat (config/config.yaml).
# ---------------------------------------------------------------

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    MAIN_LANE,
    data_footer,
    load_report,
    require_reports,
    show_chart,
)

from hff.charts import LANE_COLORS, LANE_LABELS, TEXT_SECONDARY

require_reports(
    [
        "scenario_grid.csv",
        "scenarios.csv",
        "scenario_assumptions.csv",
        "sarimax_coefficients.csv",
    ]
)
grid = load_report("scenario_grid.csv")
named = load_report("scenarios.csv")
assumptions = load_report("scenario_assumptions.csv").set_index("scenario")
latest = assumptions.loc["status_quo"]
color = LANE_COLORS[MAIN_LANE]

coefficients = load_report("sarimax_coefficients.csv")
lane_coefficients = coefficients[coefficients["lane_id"] == MAIN_LANE]
significant = int((lane_coefficients["p_value"] < 0.05).sum())

st.title("Scenario explorer")
st.warning(
    "Conditional results, not forecasts. For this lane, "
    f"{significant} of the {len(lane_coefficients)} SARIMAX driver effects are "
    "statistically different from zero (p < 0.05), so compare the scenarios with "
    "the width of the 80% range before reading anything into them."
)
st.markdown(
    f"Latest driver values: Brent USD {latest['brent_usd_latest']:.0f} per barrel, "
    f"risk index {latest['gprd_latest']:.0f}, Gulf container port calls "
    f"{latest['gulf_calls_latest']:.0f} per week."
)

columns = st.columns(3)
with columns[0]:
    brent = st.select_slider(
        "Brent in 8 weeks, USD per barrel",
        sorted(grid["brent_usd_target"].unique()),
        value=sorted(grid["brent_usd_target"].unique())[-2],
    )
with columns[1]:
    gprd = st.select_slider(
        "Geopolitical risk index in 8 weeks", sorted(grid["gprd_target"].unique())
    )
with columns[2]:
    calls = st.select_slider(
        "Gulf container port calls per week in 8 weeks",
        sorted(grid["gulf_calls_target"].unique()),
    )

chosen = grid[
    (grid["brent_usd_target"] == brent)
    & (grid["gprd_target"] == gprd)
    & (grid["gulf_calls_target"] == calls)
]
status_quo = named[named["scenario"] == "status_quo"]

# ---------------------------------------------------------------
# 1. Fan chart of the chosen path, status quo for reference
# ---------------------------------------------------------------
tooltip = [
    alt.Tooltip("horizon:Q", title="Weeks ahead"),
    alt.Tooltip("median_change_pct:Q", title="Median change, %", format="+.1f"),
    alt.Tooltip("low_change_pct:Q", title="80% range from, %", format="+.1f"),
    alt.Tooltip("high_change_pct:Q", title="80% range to, %", format="+.1f"),
]
base = alt.Chart(chosen).encode(
    x=alt.X("horizon:Q", title="Weeks ahead", scale=alt.Scale(domain=[1, 12]))
)
band = base.mark_area(color=color, opacity=0.18).encode(
    y=alt.Y("low_change_pct:Q", title="Change from the latest week, %"),
    y2="high_change_pct:Q",
)
median = base.mark_line(color=color, strokeWidth=2, point=True).encode(
    y="median_change_pct:Q", tooltip=tooltip
)
reference = (
    alt.Chart(status_quo)
    .mark_line(color=TEXT_SECONDARY, strokeDash=[4, 3])
    .encode(x="horizon:Q", y="median_change_pct:Q", tooltip=tooltip)
)
st.subheader(f"{LANE_LABELS[MAIN_LANE]}: rate change implied by the chosen levels")
show_chart(band + reference + median, 380)
final = chosen[chosen["horizon"] == chosen["horizon"].max()].iloc[0]
st.caption(
    f"After 12 weeks: median {final['median_change_pct']:+.0f}%, 80% range "
    f"{final['low_change_pct']:+.0f}% to {final['high_change_pct']:+.0f}%. "
    "Dashed grey: the status quo median."
)

# ---------------------------------------------------------------
# 2. The three named scenarios
# ---------------------------------------------------------------
st.subheader("Named scenarios after 12 weeks")
scenario_names = {
    "reopening": "Reopening: drivers back to their pre-shock average",
    "status_quo": "Status quo: drivers stay at their latest values",
    "escalation": "Escalation: port calls halve, Brent +25%, risk +30%",
}
last_horizon = named["horizon"].max()
table_rows = []
for row in named[named["horizon"] == last_horizon].itertuples():
    table_rows.append(
        {
            "Scenario": scenario_names[row.scenario],
            "Median, %": f"{row.median_change_pct:+.0f}",
            "80% range, %": f"{row.low_change_pct:+.0f} to {row.high_change_pct:+.0f}",
        }
    )
st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
data_footer()
