# ---------------------------------------------------------------
# Page: Pricing illustration
# Purpose: Each week a carrier on the Middle East lane can lock today's
#          rate for 4 weeks or stay on the spot market. The page shows
#          the value of simple decision rules over the backtest.
# Input:   reports/pricing_policy.csv
#
# Value of locking in week t, in percent of today's rate:
#   value_lock[t] = 100 * (r[t] - mean(r[t+1..t+4])) / r[t]
# positive when spot rates then fall. A forecast policy locks when the
# model's 4-week median is below today's rate. Every value is measured
# against always staying on spot.
# ---------------------------------------------------------------

import altair as alt
import streamlit as st
from common import (
    PERIOD_LABELS,
    data_footer,
    load_report,
    require_reports,
    show_chart,
)

from hff.charts import BETTER, WORSE

POLICY_NAMES = {
    "perfect_foresight": "Perfect foresight (upper bound)",
    "forecast_sarimax": "Lock when SARIMAX forecasts a fall",
    "forecast_arima": "Lock when ARIMA forecasts a fall",
    "forecast_lightgbm": "Lock when LightGBM forecasts a fall",
    "forecast_drift": "Lock when drift forecasts a fall",
    "always_lock": "Always lock",
}

require_reports(["pricing_policy.csv"])
policies = load_report("pricing_policy.csv")

st.title("Pricing illustration: lock or stay on spot")
st.markdown(
    "An illustration of business value, not a pricing engine. Weeks are treated "
    "as independent decisions; volumes, contract terms and customer reactions "
    "are ignored."
)
period = st.selectbox(
    "Period", list(PERIOD_LABELS), format_func=PERIOD_LABELS.get, key="pricing_period"
)
view = policies[policies["period"] == period].copy()
view["Policy"] = view["policy"].map(POLICY_NAMES)
view["direction"] = "gain"
view.loc[view["mean_value_vs_spot_pct"] < 0, "direction"] = "loss"
view["locked_pct"] = 100 * view["share_of_weeks_locked"]

bars = (
    alt.Chart(view)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X(
            "mean_value_vs_spot_pct:Q",
            title="Mean value per weekly decision vs always spot, % of the rate",
        ),
        y=alt.Y("Policy:N", sort=list(POLICY_NAMES.values()), title=None),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["gain", "loss"], range=[BETTER, WORSE]),
            legend=None,
        ),
        tooltip=[
            "Policy:N",
            alt.Tooltip("mean_value_vs_spot_pct:Q", title="Value, %", format="+.2f"),
            alt.Tooltip("locked_pct:Q", title="Weeks locked, %", format=".0f"),
            alt.Tooltip("weeks:Q", title="Weeks"),
        ],
    )
)
# Values are written just outside the end of each bar: to the right of
# gains, to the left of losses.
label_base = alt.Chart(view).encode(
    x=alt.X("mean_value_vs_spot_pct:Q"),
    y=alt.Y("Policy:N", sort=list(POLICY_NAMES.values())),
    text=alt.Text("mean_value_vs_spot_pct:Q", format="+.1f"),
)
gain_labels = label_base.transform_filter(alt.datum.direction == "gain").mark_text(
    color="#0b0b0b", fontSize=12, dx=6, align="left"
)
loss_labels = label_base.transform_filter(alt.datum.direction == "loss").mark_text(
    color="#0b0b0b", fontSize=12, dx=-6, align="right"
)
st.subheader(f"Value of each rule, {PERIOD_LABELS[period].lower()}")
show_chart(bars + gain_labels + loss_labels, 300)
st.caption(
    "Blue: better than always staying on spot; red: worse. Locking pays only "
    "when rates fall afterwards, so it loses during a rising shock."
)
data_footer()
