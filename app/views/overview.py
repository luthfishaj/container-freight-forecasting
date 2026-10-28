# ---------------------------------------------------------------
# Page: Overview
# Purpose: The findings on one screen: headline numbers, the four
#          lanes around the Hormuz closure, and the full rate history.
# Input:   reports/eda_hormuz.csv, rebased_rates.csv, leaderboard.csv,
#          interval_metrics.csv; data/events.csv
#
# Rebasing: a lane's index at week t divided by its index at the
# reference week, times 100. A value of 250 means the rate is 2.5
# times its reference level (+150%).
# ---------------------------------------------------------------

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    MAIN_LANE,
    data_footer,
    end_labels,
    lane_scale,
    load_events,
    load_report,
    require_reports,
    shock_bands,
    show_chart,
)

from hff.charts import LANE_LABELS, TEXT_MUTED

require_reports(
    ["eda_hormuz.csv", "rebased_rates.csv", "leaderboard.csv", "interval_metrics.csv"]
)
hormuz = load_report("eda_hormuz.csv").set_index("lane_id")
rates = load_report("rebased_rates.csv")
rates["week_date"] = pd.to_datetime(rates["week_date"])
rates["lane"] = rates["lane_id"].map(LANE_LABELS)
lane_ids = list(hormuz.index)
leaderboard = load_report("leaderboard.csv")
intervals = load_report("interval_metrics.csv")
events = load_events()
closure_date = events[events["category"] == "hormuz"]["date"].min()

st.title("Container spot rates through the Hormuz closure")
st.markdown(
    "Weekly spot rates from Ningbo to the Middle East (Dammam and Dubai), "
    "forecast 1 to 12 weeks ahead with 80% ranges, and compared with three "
    "lanes less exposed to the Gulf."
)

# ---------------------------------------------------------------
# 1. Headline numbers
# ---------------------------------------------------------------
main = hormuz.loc[MAIN_LANE]
reference_week = pd.Timestamp(main["reference_week"])
overall = leaderboard[
    (leaderboard["period"] == "all") & (leaderboard["lane_id"] == MAIN_LANE)
]
mase = overall.pivot(index="model", columns="horizon", values="mase")
ratio = mase.drop(index=["naive", "seasonal_naive"]) / mase.loc["naive"]
best_short = ratio[1].idxmin()
best_long = float(ratio[[4, 8, 12]].min().min())
widened = intervals[
    (intervals["period"] == "all")
    & (intervals["lane_id"] == MAIN_LANE)
    & (intervals["model"] == "lightgbm_widened")
]["coverage_80"]

columns = st.columns(4)
columns[0].metric(
    f"Middle East since {reference_week:%d %b %Y}",
    f"{main['change_to_latest_pct']:+.0f}%",
    f"peak {main['peak_change_pct']:+.0f}%",
    delta_color="off",
)
columns[1].metric(
    "US West Coast, same period",
    f"{hormuz.loc['asia_us_west_coast', 'change_to_latest_pct']:+.0f}%",
    f"North Europe {hormuz.loc['asia_north_europe', 'change_to_latest_pct']:+.0f}%",
    delta_color="off",
)
columns[2].metric(
    "1-week error vs no change",
    f"{100 * (ratio.loc[best_short, 1] - 1):+.0f}%",
    f"best model: {best_short.upper()}",
    delta_color="off",
)
columns[3].metric(
    "80% ranges that held",
    f"{100 * widened.min():.0f}–{100 * widened.max():.0f}%",
    "widened LightGBM, all horizons",
    delta_color="off",
)

# ---------------------------------------------------------------
# 2. The four lanes around the closure
# ---------------------------------------------------------------
st.subheader("Rates around the Hormuz closure")
window = rates[rates["week_date"] >= reference_week - pd.Timedelta(weeks=16)].copy()
reference_values = window[window["week_date"] == reference_week].set_index("lane_id")
window["index"] = (
    100
    * window["index_first_week_100"]
    / window["lane_id"].map(reference_values["index_first_week_100"])
)
window["change_pct"] = window["index"] - 100
lines = (
    alt.Chart(window)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("week_date:T", title=None),
        y=alt.Y("index:Q", title=f"Index, week of {reference_week:%d %b %Y} = 100"),
        color=alt.Color("lane:N", scale=lane_scale(lane_ids), title=None),
        tooltip=[
            alt.Tooltip("lane:N", title="Lane"),
            alt.Tooltip("week_date:T", title="Week ending", format="%d %b %Y"),
            alt.Tooltip(
                "change_pct:Q", title="Change since reference, %", format="+.0f"
            ),
        ],
    )
)
closure = pd.DataFrame({"date": [closure_date], "label": ["Hormuz closure"]})
closure_rule = alt.Chart(closure).mark_rule(color=TEXT_MUTED).encode(
    x="date:T"
) + alt.Chart(closure).mark_text(
    align="left", baseline="top", dx=4, color=TEXT_MUTED, fontSize=11
).encode(x="date:T", y=alt.value(0), text="label:N")
last_points = window.sort_values("week_date").groupby("lane_id").tail(1).copy()
last_points["x"] = last_points["week_date"]
last_points["label"] = (
    last_points["lane"] + " " + last_points["change_pct"].map("{:+.0f}%".format)
)
min_gap = 0.05 * (window["index"].max() - window["index"].min())
show_chart(closure_rule + lines + end_labels(last_points, "index", min_gap), 380)
st.caption(
    "The Middle East index rose "
    f"{main['change_week_before_closure_pct']:+.0f}% in the week before the "
    "closure date, so the reference is the week before that rise."
)

# ---------------------------------------------------------------
# 3. Full history
# ---------------------------------------------------------------
st.subheader("Seven years of spot rates (log scale)")
history = (
    alt.Chart(rates)
    .mark_line(strokeWidth=1.5)
    .encode(
        x=alt.X("week_date:T", title=None),
        y=alt.Y(
            "index_first_week_100:Q",
            title="Index, first week = 100 (log scale)",
            scale=alt.Scale(type="log"),
        ),
        color=alt.Color("lane:N", scale=lane_scale(lane_ids), title=None),
        tooltip=[
            alt.Tooltip("lane:N", title="Lane"),
            alt.Tooltip("week_date:T", title="Week ending", format="%d %b %Y"),
            alt.Tooltip("index_first_week_100:Q", title="Index", format=".0f"),
            alt.Tooltip("weekly_change_pct:Q", title="Weekly change, %", format="+.1f"),
        ],
    )
)
bands = shock_bands(rates["week_date"].min(), rates["week_date"].max())
show_chart(bands + history, 380)

# ---------------------------------------------------------------
# 4. Findings
# ---------------------------------------------------------------
st.subheader("What the results show")
st.markdown(
    f"""
- **The Gulf lane moved first and most:** {main["change_to_latest_pct"]:+.0f}% since
  the week of {reference_week:%d %B}, with a peak of {main["peak_change_pct"]:+.0f}%
  in the week of {pd.Timestamp(main["peak_week"]):%d %B %Y}.
- **The US West Coast rose almost as much**
  ({hormuz.loc["asia_us_west_coast", "change_to_latest_pct"]:+.0f}%), while North
  Europe and the West Mediterranean peaked in June and fell back.
- **Short horizons can be forecast:** one week ahead, {best_short.upper()} errors
  are {100 * (1 - ratio.loc[best_short, 1]):.0f}% smaller than "no change".
  From 4 weeks on, the best model reaches only {best_long:.2f} times the error of
  "no change" (see *Model accuracy*).
- **The ranges are usable:** after a correction from past misses, 80% ranges
  contain {100 * widened.min():.0f}–{100 * widened.max():.0f}% of the outcomes.
- **The shock itself was not foreseen** by any model (see *Hormuz stress test*),
  and the public drivers add no measurable accuracy.
"""
)
data_footer()
