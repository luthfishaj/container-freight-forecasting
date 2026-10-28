# ---------------------------------------------------------------
# Module: app/common.py
# Purpose: Shared pieces of the dashboard: loading the result files,
#          labels and colours, and small chart helpers (Altair).
#
# The dashboard reads only public files: reports/ (percent changes,
# rebased indices and aggregate metrics, D-022), config/config.yaml
# and data/events.csv. It never reads the raw or processed data, so it
# can be deployed without republishing the NCFI values.
#
# Every chart has hover tooltips; charts with several series have a
# legend; colours follow hff/charts.py (lanes, models, shock bands).
# ---------------------------------------------------------------

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from hff.charts import (
    AXIS,
    GRID,
    LANE_COLORS,
    LANE_LABELS,
    MODEL_COLORS,
    MODEL_LABELS,
    SHOCK_BAND,
    SHOCK_LABELS,
    TEXT_MUTED,
    TEXT_SECONDARY,
)
from hff.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
CONFIG = load_config()
MAIN_LANE = [lane["id"] for lane in CONFIG["lanes"] if lane["role"] == "main"][0]
LANE_IDS = list(LANE_COLORS)
PERIOD_LABELS = {
    "all": "All periods",
    "calm": "Calm",
    "covid": "COVID",
    "red_sea": "Red Sea",
    "hormuz": "Hormuz",
}


@st.cache_data
def load_report(file_name: str) -> pd.DataFrame:
    """Reads one CSV file of reports/ (cached between page views)."""
    return pd.read_csv(REPORTS / file_name)


@st.cache_data
def load_events() -> pd.DataFrame:
    """The dated event calendar, each event with its source."""
    return pd.read_csv(PROJECT_ROOT / "data" / "events.csv", parse_dates=["date"])


def require_reports(file_names: list[str]) -> None:
    """Stops the page with a clear message if a result file is missing."""
    missing = []
    for file_name in file_names:
        if not (REPORTS / file_name).exists():
            missing.append(file_name)
    if missing:
        st.error(
            "Missing result files: "
            + ", ".join(missing)
            + ". Run `make all` (or the step that writes them) first."
        )
        st.stop()


def lane_picker(label: str = "Lane", key: str | None = None) -> str:
    """Select box of the four lanes, main lane first; returns the lane id."""
    return st.selectbox(label, LANE_IDS, format_func=LANE_LABELS.get, index=0, key=key)


def lane_scale(lane_ids: list[str]) -> alt.Scale:
    """Colour scale that keeps every lane in its fixed colour."""
    labels = []
    colors = []
    for lane_id in lane_ids:
        labels.append(LANE_LABELS[lane_id])
        colors.append(LANE_COLORS[lane_id])
    return alt.Scale(domain=labels, range=colors)


def model_scale(models: list[str]) -> alt.Scale:
    """Colour scale that keeps every model in its fixed colour."""
    labels = []
    colors = []
    for model in models:
        labels.append(MODEL_LABELS[model])
        colors.append(MODEL_COLORS[model])
    return alt.Scale(domain=labels, range=colors)


def shock_bands(first_date: pd.Timestamp, last_date: pd.Timestamp) -> alt.Chart:
    """Light grey bands for the shock periods inside the date range."""
    rows = []
    for name, period in CONFIG["shock_periods"].items():
        start = pd.Timestamp(period["start"])
        end = last_date if period["end"] is None else pd.Timestamp(period["end"])
        if end < first_date or start > last_date:
            continue
        rows.append(
            {
                "start": max(start, first_date),
                "end": min(end, last_date),
                "shock": SHOCK_LABELS.get(name, name),
            }
        )
    bands = pd.DataFrame(rows)
    if bands.empty:
        bands = pd.DataFrame(columns=["start", "end", "shock"])
    base = alt.Chart(bands)
    rectangles = base.mark_rect(color=SHOCK_BAND).encode(
        x="start:T", x2="end:T", tooltip=[alt.Tooltip("shock:N", title="Period")]
    )
    labels = base.mark_text(
        align="left", baseline="top", dx=3, dy=3, color=TEXT_MUTED, fontSize=11
    ).encode(x="start:T", y=alt.value(0), text="shock:N")
    return rectangles + labels


def end_labels(last_points: pd.DataFrame, y_column: str, min_gap: float) -> alt.Chart:
    """Direct labels at the right end of each line, moved apart so that
    two labels are never closer than min_gap (in the units of y_column).

    last_points has one row per line with columns x, y_column, label and
    lane (for the colour of the dot).
    """
    ordered = last_points.sort_values(y_column).reset_index(drop=True)
    label_positions = []
    previous = None
    for value in ordered[y_column]:
        if previous is not None and value < previous + min_gap:
            value = previous + min_gap
        label_positions.append(value)
        previous = value
    ordered["label_y"] = label_positions
    base = alt.Chart(ordered)
    dots = base.mark_point(filled=True, size=40).encode(
        x="x:T",
        y=f"{y_column}:Q",
        color=alt.Color("lane:N", legend=None),
    )
    texts = base.mark_text(align="left", dx=8, fontSize=12, color="#0b0b0b").encode(
        x="x:T", y="label_y:Q", text="label:N"
    )
    return dots + texts


def styled(chart: alt.Chart, height: int = 360) -> alt.Chart:
    """Applies the project style: hairline grid, muted axes, no frame."""
    return (
        chart.properties(height=height)
        .configure_axis(
            gridColor=GRID,
            domainColor=AXIS,
            tickColor=AXIS,
            labelColor=TEXT_SECONDARY,
            titleColor=TEXT_SECONDARY,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=TEXT_SECONDARY,
            titleColor=TEXT_SECONDARY,
            orient="top",
            labelFontSize=12,
        )
        .configure_view(strokeWidth=0)
    )


def show_chart(chart: alt.Chart, height: int = 360) -> None:
    """Draws a styled chart across the page width."""
    st.altair_chart(styled(chart, height), use_container_width=True)


def data_footer() -> None:
    """Source line shown at the bottom of every page."""
    rates = load_report("rebased_rates.csv")
    st.caption(
        f"Data to the week ending {rates['week_date'].max()}. Rates: Ningbo "
        "Containerized Freight Index (Ningbo Shipping Exchange), shown as "
        "percent changes or rebased indices only. Drivers: FRED (Brent), "
        "Caldara and Iacoviello (geopolitical risk), IMF PortWatch (port calls). "
        "Full method and sources: reports/results_report.md."
    )
