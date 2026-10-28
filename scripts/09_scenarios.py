# ---------------------------------------------------------------
# Script: 09_scenarios.py
# Purpose: Three conditional scenarios for the main lane over the
#          next 12 weeks. Each scenario is an assumed path for the
#          drivers; the SARIMAX model of Phase 6 turns each path into
#          a rate path with an 80% interval.
#          These are "what if" calculations, NOT predictions: they say
#          what the model implies IF the drivers follow the path.
# Input:   data/processed/panel.parquet, config/config.yaml
# Output:  reports/scenario_assumptions.csv  driver values per scenario
#          reports/scenarios.csv             rate change per scenario
#                                            and horizon, in percent
#          reports/figures/scenarios.png
#          reports/scenario_grid.csv         100 driver paths for the
#                                            dashboard's explorer
#          MLflow experiment "business", run "scenarios"
#
# The rate is shown as a change in percent from the latest known week,
# so no NCFI value is published (D-022). The model is the SARIMAX of
# 06_backtest_models.py: ARIMA(1,1,0) on y with x = log Brent,
# log GPR and log(1 + Gulf port calls), fitted on every known week.
# ---------------------------------------------------------------

import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from hff.charts import LANE_COLORS, LANE_LABELS, SHOCK_BAND, TEXT_MUTED, apply_style
from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel
from hff.models import driver_regressors, fit_sarimax
from hff.tracking import log_finite_metrics, start_tracking

config = load_config()
settings = config["scenarios"]
reports_folder = PROJECT_ROOT / "reports"
figures_folder = PROJECT_ROOT / config["paths"]["figures"]
warnings.filterwarnings("ignore")
apply_style()

DRIVERS = ["brent_usd", "gprd", "gulf_container_calls"]
SCENARIO_LABELS = {
    "reopening": "Reopening",
    "status_quo": "Status quo",
    "escalation": "Escalation",
}
horizon_weeks = max(config["horizons"])

# ---------------------------------------------------------------
# 1. Main lane history and fitted SARIMAX
# ---------------------------------------------------------------
main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
panel = load_panel(config)
lane = panel[(panel["lane_id"] == main_lane) & panel["y_known"].notna()]
lane = lane.reset_index(drop=True)
lane[DRIVERS] = lane[DRIVERS].ffill().bfill()


def driver_logs(drivers: pd.DataFrame) -> np.ndarray:
    """The model's regressors: log Brent, log GPR, log(1 + port calls)."""
    return driver_regressors(drivers).to_numpy()


fitted = fit_sarimax(
    lane["y_known"].to_numpy(), driver_logs(lane[DRIVERS]), config["arima_order"]
)

# ---------------------------------------------------------------
# 2. Driver paths
# ---------------------------------------------------------------
latest = lane[DRIVERS].iloc[-1]
latest_week = lane["week_date"].iloc[-1]
hormuz_start = pd.Timestamp(config["shock_periods"]["hormuz"]["start"])
before_shock = lane[lane["week_date"] < hormuz_start].tail(
    settings["pre_shock_reference_weeks"]
)
pre_shock = before_shock[DRIVERS].mean()

escalation = settings["escalation"]
targets = {
    "reopening": pre_shock,
    "status_quo": latest,
    "escalation": pd.Series(
        {
            "brent_usd": latest["brent_usd"] * escalation["brent_usd_factor"],
            "gprd": latest["gprd"] * escalation["gprd_factor"],
            "gulf_container_calls": latest["gulf_container_calls"]
            * escalation["gulf_container_calls_factor"],
        }
    ),
}


def driver_path(target: pd.Series) -> pd.DataFrame:
    """Straight line from the latest driver values to the target, then flat."""
    steps = np.arange(1, horizon_weeks + 1)
    share_of_move = np.minimum(steps / settings["weeks_to_target"], 1.0)
    path = pd.DataFrame(index=steps)
    for driver in DRIVERS:
        path[driver] = latest[driver] + share_of_move * (
            target[driver] - latest[driver]
        )
    return path


assumption_rows = []
paths = {}
for scenario, target in targets.items():
    paths[scenario] = driver_path(target)
    assumption_rows.append(
        {
            "scenario": scenario,
            "brent_usd_latest": round(latest["brent_usd"], 1),
            "brent_usd_target": round(target["brent_usd"], 1),
            "gprd_latest": round(latest["gprd"], 1),
            "gprd_target": round(target["gprd"], 1),
            "gulf_calls_latest": round(latest["gulf_container_calls"], 1),
            "gulf_calls_target": round(target["gulf_container_calls"], 1),
            "weeks_to_target": settings["weeks_to_target"],
        }
    )
pd.DataFrame(assumption_rows).to_csv(
    reports_folder / "scenario_assumptions.csv", index=False
)

# ---------------------------------------------------------------
# 3. Rate paths, as percent change from the latest known week
# ---------------------------------------------------------------
y_latest = lane["y_known"].iloc[-1]


def rate_change_rows(path: pd.DataFrame) -> list[dict]:
    """SARIMAX median and 80% interval for one driver path, per horizon,
    in percent change from the latest known week."""
    prediction = fitted.get_forecast(horizon_weeks, exog=driver_logs(path))
    interval = prediction.conf_int(alpha=0.2)
    rows = []
    for step in range(horizon_weeks):
        rows.append(
            {
                "horizon": step + 1,
                "week_date": (latest_week + pd.Timedelta(weeks=step + 1)).date(),
                "median_change_pct": 100
                * (np.exp(prediction.predicted_mean[step] - y_latest) - 1),
                "low_change_pct": 100 * (np.exp(interval[step, 0] - y_latest) - 1),
                "high_change_pct": 100 * (np.exp(interval[step, 1] - y_latest) - 1),
            }
        )
    return rows


scenario_rows = []
for scenario, path in paths.items():
    for row in rate_change_rows(path):
        scenario_rows.append({"scenario": scenario} | row)
scenarios = pd.DataFrame(scenario_rows).round(1)
scenarios.to_csv(reports_folder / "scenarios.csv", index=False)

# Explorer grid for the dashboard: every combination of the target
# values in config/config.yaml (scenarios: explorer_grid).
grid = settings["explorer_grid"]
grid_rows = []
for brent_target in grid["brent_usd"]:
    for gprd_target in grid["gprd"]:
        for calls_target in grid["gulf_container_calls"]:
            target = pd.Series(
                {
                    "brent_usd": brent_target,
                    "gprd": gprd_target,
                    "gulf_container_calls": calls_target,
                }
            )
            targets_used = {
                "brent_usd_target": brent_target,
                "gprd_target": gprd_target,
                "gulf_calls_target": calls_target,
            }
            for row in rate_change_rows(driver_path(target)):
                grid_rows.append(targets_used | row)
pd.DataFrame(grid_rows).round(1).to_csv(
    reports_folder / "scenario_grid.csv", index=False
)

coefficients = pd.Series(
    fitted.params[:3], index=["log Brent", "log GPR", "log(1 + Gulf calls)"]
)
print(f"SARIMAX driver coefficients on {main_lane}:")
print(coefficients.round(3).to_string())
print(
    "Scenario rate change after 12 weeks (% from latest week, median and 80% interval):"
)
print(scenarios[scenarios["horizon"] == horizon_weeks].to_string(index=False))

# ---------------------------------------------------------------
# 4. Figure: one panel per scenario, shared scale
# ---------------------------------------------------------------
figure, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
for axis, scenario in zip(axes, SCENARIO_LABELS):
    view = scenarios[scenarios["scenario"] == scenario]
    axis.fill_between(
        view["horizon"],
        view["low_change_pct"],
        view["high_change_pct"],
        color=SHOCK_BAND,
        linewidth=0,
    )
    axis.plot(view["horizon"], view["median_change_pct"], color=LANE_COLORS[main_lane])
    axis.axhline(0, color=TEXT_MUTED, linewidth=0.6)
    final = view.iloc[-1]
    axis.annotate(
        f"{final['median_change_pct']:+.0f}%",
        (final["horizon"], final["median_change_pct"]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
    )
    axis.set_title(SCENARIO_LABELS[scenario], fontsize=9)
    axis.set_xticks([1, 4, 8, 12])
    axis.set_xlabel("Weeks ahead")
axes[0].set_ylabel(f"Change from week of {latest_week:%d %b %Y}, %")
figure.suptitle(
    f"{LANE_LABELS[main_lane]}: conditional scenarios (median and 80% interval)",
    x=0.06,
    ha="left",
    fontweight="bold",
)
figure.savefig(figures_folder / "scenarios.png", dpi=150, bbox_inches="tight")
plt.close(figure)
print("Saved reports/scenarios.csv and reports/figures/scenarios.png")

# ---------------------------------------------------------------
# 5. MLflow: assumptions, 12-week outcomes and files
# ---------------------------------------------------------------
start_tracking("business")
with mlflow.start_run(run_name="scenarios"):
    mlflow.log_params(
        {
            "lane": main_lane,
            "latest_week": latest_week.date(),
            "weeks_to_target": settings["weeks_to_target"],
            "escalation": settings["escalation"],
        }
    )
    final = scenarios[scenarios["horizon"] == horizon_weeks]
    scenario_metrics = {}
    for row in final.itertuples():
        scenario_metrics[f"median_change_{row.scenario}_h{horizon_weeks}"] = (
            row.median_change_pct
        )
        scenario_metrics[f"low_change_{row.scenario}_h{horizon_weeks}"] = (
            row.low_change_pct
        )
        scenario_metrics[f"high_change_{row.scenario}_h{horizon_weeks}"] = (
            row.high_change_pct
        )
    log_finite_metrics(scenario_metrics)
    for file_name in ["scenario_assumptions.csv", "scenarios.csv", "scenario_grid.csv"]:
        mlflow.log_artifact(str(reports_folder / file_name))
    mlflow.log_artifact(str(figures_folder / "scenarios.png"))
print("Logged the run to MLflow (experiment: business)")
