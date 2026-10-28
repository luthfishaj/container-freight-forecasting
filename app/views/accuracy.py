# ---------------------------------------------------------------
# Page: Model accuracy
# Purpose: How every model did in the rolling-origin backtest, against
#          the naive forecast ("no change"), by lane, horizon and period;
#          how well the 80% ranges held; what drives the models.
# Input:   reports/leaderboard.csv, interval_metrics.csv,
#          feature_importance.csv, sarimax_coefficients.csv
#
# MASE ratio = MASE of the model / MASE of the naive forecast, same
# lane, horizon and period. Below 1: smaller errors than "no change".
# Coverage: share of actual values inside the 80% range (target 80%).
# ---------------------------------------------------------------

import altair as alt
import pandas as pd
import streamlit as st
from common import (
    PERIOD_LABELS,
    data_footer,
    lane_picker,
    load_report,
    require_reports,
    show_chart,
)

from hff.charts import BETTER, DRIVER_INK, MODEL_COLORS, SHOCK_BAND, WORSE

MODEL_NAMES = {
    "naive": "Naive (no change)",
    "seasonal_naive": "Seasonal naive",
    "drift": "Drift",
    "ets": "ETS (damped)",
    "arima": "ARIMA",
    "sarimax": "SARIMAX",
    "lightgbm": "LightGBM",
}
INTERVAL_NAMES = {
    "lightgbm": "LightGBM as trained",
    "lightgbm_widened": "LightGBM widened",
    "sarimax": "SARIMAX",
    "naive_empirical": "Naive + past errors",
}

require_reports(
    [
        "leaderboard.csv",
        "interval_metrics.csv",
        "feature_importance.csv",
        "sarimax_coefficients.csv",
    ]
)
leaderboard = load_report("leaderboard.csv")
intervals = load_report("interval_metrics.csv")

st.title("Model accuracy")
st.markdown(
    "Every model was refitted each Friday on the weeks published so far and "
    "forecast 1 to 12 weeks ahead (rolling-origin backtest, March 2020 to "
    'October 2026). The naive forecast, "no change", is the benchmark.'
)
left, right = st.columns(2)
with left:
    lane_id = lane_picker()
with right:
    period = st.selectbox("Period", list(PERIOD_LABELS), format_func=PERIOD_LABELS.get)

# ---------------------------------------------------------------
# 1. MASE relative to the naive forecast
# ---------------------------------------------------------------
view = leaderboard[
    (leaderboard["lane_id"] == lane_id) & (leaderboard["period"] == period)
]
naive_mase = view[view["model"] == "naive"].set_index("horizon")["mase"]
view = view.copy()
view["ratio"] = view["mase"] / view["horizon"].map(naive_mase)
view["Model"] = view["model"].map(MODEL_NAMES)
view["Horizon"] = view["horizon"].astype(str) + " wk"
horizon_order = []
for horizon in sorted(view["horizon"].unique()):
    horizon_order.append(f"{horizon} wk")

st.subheader('Error relative to "no change" (MASE ratio)')
base = alt.Chart(view).encode(
    x=alt.X("Horizon:N", sort=horizon_order, title="Weeks ahead"),
    y=alt.Y("Model:N", sort=list(MODEL_NAMES.values()), title=None),
)
cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
    color=alt.Color(
        "ratio:Q",
        scale=alt.Scale(
            domain=[0.8, 1.0, 1.2], range=[BETTER, SHOCK_BAND, WORSE], clamp=True
        ),
        title="MASE ratio",
    ),
    tooltip=[
        "Model:N",
        "Horizon:N",
        alt.Tooltip("ratio:Q", title="MASE ratio", format=".2f"),
        alt.Tooltip("mape_pct:Q", title="Mean error, % of rate", format=".1f"),
        alt.Tooltip("n_forecasts:Q", title="Forecasts"),
    ],
)
labels = base.mark_text(fontSize=12, color="#0b0b0b").encode(
    text=alt.Text("ratio:Q", format=".2f")
)
show_chart(cells + labels, 300)
st.caption(
    'Blue: smaller errors than "no change"; red: larger. The colour scale stops '
    "at 0.8 and 1.2; the number in each cell is exact."
)

# Directional accuracy against always predicting the more frequent move.
direction = view[view["model"].isin(["arima", "sarimax", "lightgbm", "drift"])]
direction_table = direction.pivot(
    index="Model", columns="Horizon", values="directional_accuracy"
)[horizon_order]
majority = view[view["model"] == "arima"].set_index("Horizon")[
    "majority_direction_share"
]
direction_table.loc["Always the more frequent direction"] = majority[horizon_order]
st.markdown("**Share of up/down calls that were right**")
st.dataframe(
    (100 * direction_table).round(0).astype("Int64").astype(str) + "%",
    use_container_width=True,
)

# ---------------------------------------------------------------
# 2. Prediction intervals
# ---------------------------------------------------------------
st.subheader("Did the 80% ranges hold?")
interval_view = intervals[
    (intervals["lane_id"] == lane_id) & (intervals["period"] == period)
].copy()
interval_view["Method"] = interval_view["model"].map(INTERVAL_NAMES)
interval_view["coverage_pct"] = 100 * interval_view["coverage_80"]
method_colors = [
    MODEL_COLORS["lightgbm"],
    MODEL_COLORS["lightgbm"],
    MODEL_COLORS["sarimax"],
    MODEL_COLORS["naive"],
]
target_band = pd.DataFrame({"low": [75], "high": [85]})
band = alt.Chart(target_band).mark_rect(color=SHOCK_BAND).encode(y="low:Q", y2="high:Q")
coverage_lines = (
    alt.Chart(interval_view)
    .mark_line(point=True, strokeWidth=2)
    .encode(
        x=alt.X("horizon:O", title="Weeks ahead"),
        y=alt.Y(
            "coverage_pct:Q",
            title="Actual values inside the range, %",
            scale=alt.Scale(domain=[30, 100]),
        ),
        color=alt.Color(
            "Method:N",
            scale=alt.Scale(domain=list(INTERVAL_NAMES.values()), range=method_colors),
            title=None,
        ),
        strokeDash=alt.StrokeDash(
            "Method:N",
            scale=alt.Scale(
                domain=list(INTERVAL_NAMES.values()),
                range=[[4, 3], [1, 0], [1, 0], [1, 0]],
            ),
            title=None,
        ),
        tooltip=[
            "Method:N",
            alt.Tooltip("horizon:O", title="Weeks ahead"),
            alt.Tooltip("coverage_pct:Q", title="Inside the range, %", format=".0f"),
            alt.Tooltip("pinball_loss:Q", title="Pinball loss", format=".4f"),
        ],
    )
)
show_chart(band + coverage_lines, 320)
st.caption(
    "Shaded: the 75–85% target. LightGBM's ranges as trained are too narrow; "
    "the widened version corrects them from its own past misses."
)

# ---------------------------------------------------------------
# 3. What drives the models
# ---------------------------------------------------------------
st.subheader("What drives the models")
left, right = st.columns(2)
with left:
    importance = load_report("feature_importance.csv")
    horizon_column = st.selectbox(
        "LightGBM feature importance, horizon", ["h1", "h2", "h4", "h8", "h12"]
    )
    top = importance.nlargest(12, horizon_column)
    bars = (
        alt.Chart(top)
        .mark_bar(color=DRIVER_INK, cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{horizon_column}:Q", title="Share of total gain, %"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature:N", alt.Tooltip(f"{horizon_column}:Q", format=".1f")],
        )
    )
    show_chart(bars, 340)
with right:
    coefficients = load_report("sarimax_coefficients.csv")
    lane_coefficients = coefficients[coefficients["lane_id"] == lane_id]
    st.markdown("**SARIMAX driver coefficients**")
    st.dataframe(
        lane_coefficients[["driver", "coefficient", "p_value"]],
        hide_index=True,
        use_container_width=True,
    )
    significant = int((coefficients["p_value"] < 0.05).sum())
    st.markdown(
        "A p-value above 0.05 means the effect cannot be told apart from zero. "
        f"Across all lanes, {significant} of the {len(coefficients)} driver "
        "coefficients are below 0.05."
    )
data_footer()
