# ---------------------------------------------------------------
# Module: hff/charts.py
# Purpose: Shared chart style for every figure in reports/figures/,
#          so that all charts use the same colours, fonts and shading.
#
# Colour rules:
#   - each lane keeps the same colour in every chart (blue for the
#     main lane), in a fixed order checked for colour-blind safety;
#   - drivers, which are not lanes, use a neutral grey ink;
#   - shock periods are light grey bands, never a series colour.
# Lines are labelled directly at their end, so identity never relies
# on colour alone.
# ---------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHOCK_BAND = "#f0efec"
DRIVER_INK = "#52514e"

LANE_COLORS = {
    "asia_middle_east": "#2a78d6",
    "asia_north_europe": "#eb6834",
    "asia_west_mediterranean": "#1baf7a",
    "asia_us_west_coast": "#eda100",
}
LANE_LABELS = {
    "asia_middle_east": "Middle East",
    "asia_north_europe": "North Europe",
    "asia_west_mediterranean": "W. Mediterranean",
    "asia_us_west_coast": "US West Coast",
}
SHOCK_LABELS = {"covid": "COVID", "red_sea": "Red Sea", "hormuz": "Hormuz"}

# Models are not lanes, so they use other slots of the checked palette:
# neutral grey for the naive benchmark (a reference, not a series), then
# magenta, violet and green, validated together for colour-blind safety.
# Magenta is below 3:1 contrast on the surface, so model charts always
# carry a legend, tooltips or a table.
MODEL_COLORS = {
    "naive": "#52514e",
    "arima": "#e87ba4",
    "sarimax": "#6250d6",
    "lightgbm": "#008300",
}
MODEL_LABELS = {
    "naive": "Naive (no change)",
    "arima": "ARIMA",
    "sarimax": "SARIMAX",
    "lightgbm": "LightGBM",
}
# Diverging pair for "better or worse than a reference": blue below,
# red above, neutral grey at the reference itself.
BETTER = "#2a78d6"
WORSE = "#e34948"


def apply_style() -> None:
    """Sets the shared matplotlib style: light surface, hairline grid."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            # Grid lines stay behind bars and lines.
            "axes.axisbelow": True,
        }
    )


def shade_shock_periods(axis, shock_periods: dict, last_date: pd.Timestamp) -> None:
    """Draws one light band per shock period, labelled at the top."""
    for name, period in shock_periods.items():
        start = pd.Timestamp(period["start"])
        end = last_date if period["end"] is None else pd.Timestamp(period["end"])
        axis.axvspan(start, end, color=SHOCK_BAND, zorder=0, linewidth=0)
        axis.text(
            start,
            1.0,
            " " + SHOCK_LABELS.get(name, name),
            transform=axis.get_xaxis_transform(),
            va="top",
            ha="left",
            color=TEXT_MUTED,
            fontsize=8,
        )


# Smallest vertical distance between two end labels, as a share of the
# axis height (0.05 = 5%, enough for one line of 8-point text).
LABEL_MIN_GAP = 0.05


def label_line_ends(axis, labels: list[dict]) -> None:
    """Writes direct labels at the right end of several lines.

    Each label is a dictionary with keys "x", "y", "text" and "color".
    Labels closer than LABEL_MIN_GAP are pushed upwards, lowest first,
    so that they never overlap. Positions are compared as a share of
    the axis height, so the rule works on linear and log scales. Call
    it after every line has been drawn (the axis limits must be final).
    """
    axis.get_ylim()  # fixes the axis limits before positions are read
    data_to_axis = axis.transData + axis.transAxes.inverted()
    # Only the height is needed; the left axis limit stands in for x
    # (dates would first need converting to numbers).
    left_limit = axis.get_xlim()[0]
    positions = []
    for index, label in enumerate(labels):
        height_share = data_to_axis.transform((left_limit, float(label["y"])))[1]
        positions.append((height_share, index))
    positions.sort()

    previous = None
    for height_share, index in positions:
        if previous is not None and height_share < previous + LABEL_MIN_GAP:
            height_share = previous + LABEL_MIN_GAP
        previous = height_share
        label = labels[index]
        axis.plot(
            [label["x"]], [label["y"]], marker="o", markersize=4, color=label["color"]
        )
        axis.annotate(
            label["text"],
            (label["x"], height_share),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=TEXT_PRIMARY,
            fontsize=8,
        )


def label_line_end(axis, x_value, y_value, text: str, color: str) -> None:
    """Writes a direct label at the right end of a line, in text ink."""
    axis.plot([x_value], [y_value], marker="o", markersize=4, color=color)
    axis.annotate(
        text,
        (x_value, y_value),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=TEXT_PRIMARY,
        fontsize=8,
    )
