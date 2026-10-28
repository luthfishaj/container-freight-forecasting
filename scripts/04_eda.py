# ---------------------------------------------------------------
# Script: 04_eda.py
# Purpose: Exploratory analysis of the weekly panel: rate history,
#          the exposure of each lane to the Hormuz disruption, the
#          drivers, their link with rate changes, and volatility.
# Input:   data/processed/panel.parquet, warehouse/hff.duckdb
# Output:  reports/figures/eda_*.png
#          reports/eda_summary.csv     per lane and period
#          reports/eda_hormuz.csv      change of each lane since the
#                                      week before the Hormuz rise
#          reports/rebased_rates.csv   weekly index per lane, 100 in
#                                      the first week (dashboard)
#
# Rates are shown as an index equal to 100 at a reference week rather
# than in NCFI points: lanes become comparable, and the publisher's
# values are not reproduced in public files.
#
# Hormuz reference week: the closure is dated 28 February 2026
# (data/events.csv), but the Middle East index of the week ending 27
# February had already risen 82%. The reference is therefore the week
# ending 20 February 2026, two weeks before the first week of the
# Hormuz period in config/config.yaml, so the whole rise is measured.
# ---------------------------------------------------------------

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

from hff.charts import (
    DRIVER_INK,
    LANE_COLORS,
    LANE_LABELS,
    SHOCK_BAND,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_style,
    label_line_ends,
    shade_shock_periods,
)
from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel

config = load_config()
figures_folder = PROJECT_ROOT / config["paths"]["figures"]
reports_folder = PROJECT_ROOT / "reports"
figures_folder.mkdir(parents=True, exist_ok=True)
shock_periods = config["shock_periods"]
hormuz_start = pd.Timestamp(shock_periods["hormuz"]["start"])
hormuz_reference_week = hormuz_start - pd.Timedelta(weeks=2)
week_before_hormuz = hormuz_start - pd.Timedelta(weeks=1)
events = pd.read_csv(PROJECT_ROOT / "data" / "events.csv", parse_dates=["date"])
closure_date = events[events["category"] == "hormuz"]["date"].min()
apply_style()

panel = load_panel(config)
# Only weeks with a published value are described (no filled weeks).
published = panel.dropna(subset=["rate"]).copy()
last_date = panel["week_date"].max()
lane_ids = list(LANE_COLORS)


def save_figure(figure, name: str) -> None:
    figure.savefig(figures_folder / name, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved reports/figures/{name}")


# ---------------------------------------------------------------
# 1. Rate history, index = 100 in the first week
# ---------------------------------------------------------------
print("Figures...")
figure, axis = plt.subplots(figsize=(10, 4.5))
shade_shock_periods(axis, shock_periods, last_date)
end_labels = []
for lane_id in lane_ids:
    lane = published[published["lane_id"] == lane_id]
    rebased = 100 * lane["rate"] / lane["rate"].iloc[0]
    axis.plot(lane["week_date"], rebased, color=LANE_COLORS[lane_id])
    end_labels.append(
        {
            "x": lane["week_date"].iloc[-1],
            "y": rebased.iloc[-1],
            "text": LANE_LABELS[lane_id],
            "color": LANE_COLORS[lane_id],
        }
    )
# Log scale: equal distances mean equal percentage changes. Ticks are
# written as plain numbers (50, 100, 200...) instead of powers of 10.
axis.set_yscale("log")
axis.set_yticks([50, 100, 200, 500, 1000])
axis.yaxis.set_major_formatter(ScalarFormatter())
axis.yaxis.set_minor_formatter(NullFormatter())
label_line_ends(axis, end_labels)
axis.set_ylabel(f"Index, first week ({published['week_date'].min():%d %b %Y}) = 100")
axis.set_title("NCFI spot rates from Ningbo by lane (log scale)")
save_figure(figure, "eda_rate_history.png")

# ---------------------------------------------------------------
# 2. Exposure gradient: index = 100 in the reference week
# ---------------------------------------------------------------
hormuz_rows = []
figure, axis = plt.subplots(figsize=(10, 4.5))
window_start = hormuz_start - pd.Timedelta(weeks=16)
axis.axvline(closure_date, color=TEXT_MUTED, linewidth=0.8)
axis.text(
    closure_date,
    1.0,
    f" Hormuz closure ({closure_date:%d %b})",
    transform=axis.get_xaxis_transform(),
    va="top",
    color=TEXT_MUTED,
    fontsize=8,
)
end_labels = []
for lane_id in lane_ids:
    lane = published[published["lane_id"] == lane_id].set_index("week_date")["rate"]
    reference_value = lane[lane.index <= hormuz_reference_week].iloc[-1]
    window = lane[lane.index >= window_start]
    rebased = 100 * window / reference_value
    axis.plot(rebased.index, rebased, color=LANE_COLORS[lane_id])
    change_now = rebased.iloc[-1] - 100
    end_labels.append(
        {
            "x": rebased.index[-1],
            "y": rebased.iloc[-1],
            "text": f"{LANE_LABELS[lane_id]} {change_now:+.0f}%",
            "color": LANE_COLORS[lane_id],
        }
    )
    after_reference = rebased[rebased.index > hormuz_reference_week]
    hormuz_rows.append(
        {
            "lane_id": lane_id,
            "reference_week": hormuz_reference_week.date(),
            "change_week_before_closure_pct": round(
                rebased[week_before_hormuz] - 100, 1
            ),
            "latest_week": rebased.index[-1].date(),
            "change_to_latest_pct": round(change_now, 1),
            "peak_change_pct": round(after_reference.max() - 100, 1),
            "peak_week": after_reference.idxmax().date(),
        }
    )
axis.axhline(100, color=TEXT_MUTED, linewidth=0.6)
label_line_ends(axis, end_labels)
axis.set_ylabel(f"Index, week of {hormuz_reference_week:%d %b %Y} = 100")
axis.set_title("Rates around the Hormuz closure: Middle East vs comparison lanes")
save_figure(figure, "eda_hormuz_exposure.png")

# ---------------------------------------------------------------
# 3. Drivers (weekly values before publication lag), small multiples
# ---------------------------------------------------------------
warehouse_path = (PROJECT_ROOT / config["paths"]["warehouse"]).as_posix()
connection = duckdb.connect(warehouse_path, read_only=True)
weekly_drivers = connection.execute(
    "SELECT * FROM weekly_drivers WHERE week_date >= ? ORDER BY week_date",
    [panel["week_date"].min()],
).df()
connection.close()
weekly_drivers["week_date"] = pd.to_datetime(weekly_drivers["week_date"])

driver_titles = {
    "brent_usd": "Brent crude, USD per barrel (weekly average)",
    "gprd": "Geopolitical Risk Index (weekly average)",
    "gulf_container_calls": "Container port calls, Jebel Ali + Dammam (per week)",
}
driver_short_names = {
    "brent_usd": "Brent crude",
    "gprd": "Geopolitical risk",
    "gulf_container_calls": "Gulf container port calls",
}
figure, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
for axis, (column, title) in zip(axes, driver_titles.items()):
    shade_shock_periods(axis, shock_periods, last_date)
    axis.plot(weekly_drivers["week_date"], weekly_drivers[column], color=DRIVER_INK)
    axis.set_title(title)
save_figure(figure, "eda_drivers.png")

# ---------------------------------------------------------------
# 4. Correlation of weekly rate changes with past driver changes
# ---------------------------------------------------------------
# corr(change of y in week t, change of x in week t-k) for k = 0..8,
# on the main lane, using log changes of every series. Bars outside
# the band +/- 2/sqrt(n) are unlikely to be zero by chance alone.
main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
main = panel[panel["lane_id"] == main_lane].set_index("week_date")["y"]
rate_change = main.diff()
drivers = weekly_drivers.set_index("week_date").reindex(main.index)
figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for axis, column in zip(axes, driver_short_names):
    # log(1 + x) keeps weeks with zero port calls usable (log 0 is
    # undefined); for prices it is almost identical to log(x).
    driver_change = np.log1p(drivers[column]).diff()
    correlations = []
    for lag in range(0, 9):
        correlations.append(rate_change.corr(driver_change.shift(lag)))
    n_weeks = int((rate_change.notna() & driver_change.notna()).sum())
    band = 2 / np.sqrt(n_weeks)
    axis.bar(range(0, 9), correlations, color=DRIVER_INK, width=0.6)
    axis.axhspan(-band, band, color=SHOCK_BAND, zorder=0)
    axis.axhline(0, color=TEXT_MUTED, linewidth=0.6)
    axis.set_title(driver_short_names[column], fontsize=9)
    axis.set_xlabel("Driver change k weeks earlier")
axes[0].set_ylabel("Correlation")
figure.suptitle(
    f"{LANE_LABELS[main_lane]}: weekly rate change vs earlier driver changes",
    x=0.06,
    y=1.04,
    ha="left",
    fontweight="bold",
)
figure.text(
    0.06,
    -0.06,
    "Shaded band: ±2/√n, n = weeks with both values. Bars inside it cannot "
    "be told apart from zero.",
    color=TEXT_SECONDARY,
    fontsize=8,
)
save_figure(figure, "eda_driver_correlation.png")

# ---------------------------------------------------------------
# 5. Volatility: 12-week standard deviation of weekly changes
# ---------------------------------------------------------------
figure, axis = plt.subplots(figsize=(10, 4))
shade_shock_periods(axis, shock_periods, last_date)
end_labels = []
for lane_id in lane_ids:
    lane = panel[panel["lane_id"] == lane_id].dropna(subset=["change_std_12w"])
    volatility = 100 * lane["change_std_12w"]
    axis.plot(lane["week_date"], volatility, color=LANE_COLORS[lane_id])
    end_labels.append(
        {
            "x": lane["week_date"].iloc[-1],
            "y": volatility.iloc[-1],
            "text": LANE_LABELS[lane_id],
            "color": LANE_COLORS[lane_id],
        }
    )
label_line_ends(axis, end_labels)
axis.set_ylabel("Std of weekly change, % (12 weeks)")
axis.set_title("Volatility of weekly rate changes by lane")
save_figure(figure, "eda_volatility.png")

# ---------------------------------------------------------------
# 6. Summary tables
# ---------------------------------------------------------------
# Weekly changes are computed on each lane's full series before the
# weeks are grouped by period, so the first week of a period keeps its
# change (the first Hormuz week holds the largest move).
published["weekly_change"] = published.groupby("lane_id")["y"].diff()
summary_rows = []
for (lane_id, period), group in published.groupby(["lane_id", "period"]):
    weekly_change = group["weekly_change"].dropna()
    # Change over a shock period: from the last week before it to its
    # last week. "Calm" is several separate stretches, so it has none.
    if period == "calm":
        change_over_period = np.nan
    else:
        lane = published[published["lane_id"] == lane_id]
        before = lane[lane["week_date"] < group["week_date"].min()]
        change_over_period = round(
            100 * (group["rate"].iloc[-1] / before["rate"].iloc[-1] - 1), 1
        )
    summary_rows.append(
        {
            "lane_id": lane_id,
            "period": period,
            "weeks_published": len(group),
            "mean_weekly_change_pct": round(100 * weekly_change.mean(), 2),
            "volatility_pct": round(100 * weekly_change.std(), 2),
            "change_over_period_pct": change_over_period,
        }
    )
summary = pd.DataFrame(summary_rows)
summary.to_csv(reports_folder / "eda_summary.csv", index=False)
hormuz_table = pd.DataFrame(hormuz_rows)
hormuz_table.to_csv(reports_folder / "eda_hormuz.csv", index=False)

# Weekly series for the dashboard's interactive charts: each lane as an
# index equal to 100 in its first published week (D-022), and the
# weekly change in percent: 100 * (exp(y[t] - y[t-1]) - 1).
published = published.sort_values(["lane_id", "week_date"])
first_rate = published.groupby("lane_id")["rate"].transform("first")
rebased_table = published[["week_date", "lane_id", "period"]].copy()
rebased_table["week_date"] = rebased_table["week_date"].dt.date
rebased_table["index_first_week_100"] = (100 * published["rate"] / first_rate).round(2)
rebased_table["weekly_change_pct"] = (
    100 * (np.exp(published["weekly_change"]) - 1)
).round(2)
rebased_table.to_csv(reports_folder / "rebased_rates.csv", index=False)

print(f"Change since the reference week {hormuz_reference_week.date()}:")
print(hormuz_table.to_string(index=False))
print("Volatility (std of weekly change, %) by lane and period:")
print(
    summary.pivot(
        index="lane_id", columns="period", values="volatility_pct"
    ).to_string()
)
