# ---------------------------------------------------------------
# Page: Hormuz stress test
# Purpose: How the models did week by week after two shocks, frozen
#          (parameters fixed before the shock) or retrained every week.
#            Hormuz:  frozen on 2026-02-27
#            Red Sea: frozen on 2023-12-08
# Input:   reports/stress_test_errors.csv, stress_test_summary.csv
#
# Error of one forecast, in percent of the actual rate:
#   100 * |forecast rate - actual rate| / actual rate
# ---------------------------------------------------------------

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    data_footer,
    lane_picker,
    lane_scale,
    load_report,
    model_scale,
    require_reports,
    show_chart,
)

from hff.charts import LANE_LABELS, MODEL_LABELS

SHOCK_NAMES = {
    "hormuz": "Hormuz (frozen 27 Feb 2026)",
    "red_sea": "Red Sea (frozen 8 Dec 2023)",
}
FIRST_WEEKS = 8
MODELS = ["naive", "arima", "sarimax", "lightgbm"]

require_reports(["stress_test_errors.csv", "stress_test_summary.csv"])
errors = load_report("stress_test_errors.csv")
errors["target_date"] = pd.to_datetime(errors["target_date"])
errors["Model"] = errors["model"].map(MODEL_LABELS)

st.title("Hormuz stress test")
st.markdown(
    "Models were frozen on the last Friday before each shock, then run through "
    "every following week, and compared with the same models refitted every week."
)
columns = st.columns(4)
with columns[0]:
    shock = st.selectbox("Shock", list(SHOCK_NAMES), format_func=SHOCK_NAMES.get)
with columns[1]:
    lane_id = lane_picker()
with columns[2]:
    horizon = st.selectbox("Weeks ahead", [1, 4])
with columns[3]:
    mode = st.radio("Models", ["retrained", "frozen"], horizontal=True)

view = errors[
    (errors["shock"] == shock)
    & (errors["lane_id"] == lane_id)
    & (errors["horizon"] == horizon)
    & (errors["mode"] == mode)
]

# ---------------------------------------------------------------
# 1. Error trajectory
# ---------------------------------------------------------------
st.subheader(f"{LANE_LABELS[lane_id]}: error of each forecast, {horizon} week(s) ahead")
lines = (
    alt.Chart(view)
    .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
    .encode(
        x=alt.X("target_date:T", title=None),
        y=alt.Y("error_pct:Q", title="Error, % of the actual rate"),
        color=alt.Color("Model:N", scale=model_scale(MODELS), title=None),
        tooltip=[
            "Model:N",
            alt.Tooltip("target_date:T", title="Target week", format="%d %b %Y"),
            alt.Tooltip("error_pct:Q", title="Error, %", format=".1f"),
        ],
    )
)
show_chart(lines, 380)

# ---------------------------------------------------------------
# 2. First weeks vs later weeks, frozen vs retrained
# ---------------------------------------------------------------
both_modes = errors[
    (errors["shock"] == shock)
    & (errors["lane_id"] == lane_id)
    & (errors["horizon"] == horizon)
]
first_target = both_modes["target_date"].min()
cutoff = first_target + pd.Timedelta(weeks=FIRST_WEEKS - 1)
summary_rows = []
for (model, model_mode), group in both_modes.groupby(["model", "mode"]):
    first = group[group["target_date"] <= cutoff]["error_pct"]
    later = group[group["target_date"] > cutoff]["error_pct"]
    summary_rows.append(
        {
            "Model": MODEL_LABELS[model],
            "Mode": model_mode,
            "First week, %": round(
                group.sort_values("target_date")["error_pct"].iloc[0], 1
            ),
            f"Weeks 1–{FIRST_WEEKS}, %": round(first.mean(), 1),
            f"After week {FIRST_WEEKS}, %": round(later.mean(), 1),
        }
    )
st.markdown("**Mean error, % of the actual rate**")
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------
# 3. How much each lane moved: naive error by lane
# ---------------------------------------------------------------
summary = load_report("stress_test_summary.csv")
naive = summary[
    (summary["shock"] == shock)
    & (summary["horizon"] == horizon)
    & (summary["model"] == "naive")
    & (summary["mode"] == "retrained")
].copy()
naive["Lane"] = naive["lane_id"].map(LANE_LABELS)
st.subheader("How hard each lane was to forecast during the shock")
bars = (
    alt.Chart(naive)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("mean_error_pct:Q", title='Mean error of "no change", % of the rate'),
        y=alt.Y("Lane:N", sort=list(LANE_LABELS.values()), title=None),
        color=alt.Color(
            "Lane:N", scale=lane_scale(list(naive["lane_id"])), legend=None
        ),
        tooltip=[
            "Lane:N",
            alt.Tooltip("mean_error_pct:Q", title="Error, %", format=".1f"),
        ],
    )
)
labels = bars.mark_text(align="left", dx=4, color="#0b0b0b").encode(
    text=alt.Text("mean_error_pct:Q", format=".0f"), color=alt.value("#0b0b0b")
)
show_chart(bars + labels, 200)
st.markdown(
    "- No model foresaw the first jump after the Hormuz closure.\n"
    "- ARIMA and SARIMAX carry recent momentum forward: that helped after the "
    "Hormuz jump, and hurt at 4 weeks after the Red Sea jump, when rates reversed.\n"
    "- Frozen and retrained models are close: the models adapt through the latest "
    "rate they start from, more than through their parameters."
)
data_footer()
