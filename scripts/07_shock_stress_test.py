# ---------------------------------------------------------------
# Script: 07_shock_stress_test.py
# Purpose: Stress test of the models on two shocks. Models are frozen
#          (their parameters fixed) on the data available just before
#          a shock, then compared week by week with the same models
#          retrained every week (from the Phase 5 and 6 backtests).
#            Hormuz:  frozen on 2026-02-27, evaluated from 2026-03-06
#            Red Sea: frozen on 2023-12-08, evaluated from 2023-12-15
# Input:   data/processed/panel.parquet
#          data/processed/forecasts_baselines.csv, forecasts_models.csv
# Output:  reports/stress_test_errors.csv  error per lane, model,
#                                          horizon, mode and shock
#          reports/stress_test_summary.csv mean error and coverage
#          reports/figures/stress_*.png
#          MLflow experiment "stress_test" (hff/tracking.py)
#
# A frozen model still sees each new published rate as input (the
# forecast starts from the latest value), but its parameters are not
# re-estimated. The comparison answers: how badly did models built on
# pre-shock data do, and how much did retraining help?
# Error of one forecast, in percent of the actual rate (the same
# definition as MAPE in hff/metrics.py):
#   error_pct = 100 * |r_hat - r| / r = 100 * |exp(y_hat - y) - 1|
# where r_hat = exp(y_hat) is the forecast rate and r the actual rate.
# ---------------------------------------------------------------

import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from hff.charts import (
    LANE_COLORS,
    LANE_LABELS,
    MODEL_COLORS,
    TEXT_MUTED,
    apply_style,
)
from hff.config import PROJECT_ROOT, load_config
from hff.io import load_panel
from hff.models import (
    SARIMAX_DRIVERS,
    add_model_features,
    fit_lightgbm,
    fit_sarimax,
    lightgbm_features,
)
from hff.tracking import log_finite_metrics, start_tracking

config = load_config()
processed_folder = PROJECT_ROOT / config["paths"]["processed_data"]
reports_folder = PROJECT_ROOT / "reports"
figures_folder = PROJECT_ROOT / config["paths"]["figures"]
warnings.filterwarnings("ignore")
apply_style()

SHOCKS = {"hormuz": "2026-02-27", "red_sea": "2023-12-08"}
SHOCK_NAMES = {"hormuz": "Hormuz", "red_sea": "Red Sea"}
# Models drawn in the error charts, in the model colours of hff.charts
# (ARIMA is left out: its errors are almost those of SARIMAX). Frozen
# models are drawn dotted, retrained ones solid.
CHART_MODELS = ["naive", "sarimax", "lightgbm"]
STRESS_HORIZONS = [1, 4]

# Same features and models as the backtests (hff.models).
panel = load_panel(config)
panel = add_model_features(panel, config, STRESS_HORIZONS)
lane_ids = sorted(panel["lane_id"].unique())
LIGHTGBM_FEATURES = lightgbm_features(config)

# ---------------------------------------------------------------
# 1. Frozen forecasts after each shock
# ---------------------------------------------------------------
frozen_rows = []
for shock, freeze_text in SHOCKS.items():
    freeze_date = pd.Timestamp(freeze_text)
    print(f"{shock}: models frozen on {freeze_date.date()}")

    # LightGBM median models trained once, on targets published before
    # the freeze date.
    frozen_lightgbm = {}
    for horizon in STRESS_HORIZONS:
        target_week = panel["week_date"] + pd.Timedelta(weeks=horizon)
        training_rows = panel[
            (target_week < freeze_date) & panel[f"target_{horizon}"].notna()
        ]
        frozen_lightgbm[horizon] = fit_lightgbm(training_rows, horizon, 0.5, config)

    for lane_id in lane_ids:
        lane = panel[panel["lane_id"] == lane_id].reset_index(drop=True)
        lane = lane[lane["y_known"].notna()].reset_index(drop=True)
        drivers = lane[SARIMAX_DRIVERS].ffill().bfill()
        before = lane["week_date"] <= freeze_date

        # ARIMA and SARIMAX parameters estimated once, before the shock.
        arima_fit = ARIMA(
            lane.loc[before, "y_known"].to_numpy(), order=tuple(config["arima_order"])
        ).fit()
        sarimax_fit = fit_sarimax(
            lane.loc[before, "y_known"].to_numpy(),
            drivers[before].to_numpy(),
            config["arima_order"],
        )

        origins = lane.index[lane["week_date"] >= freeze_date]
        for origin in origins[:-1]:
            history_y = lane.loc[:origin, "y_known"].to_numpy()
            history_x = drivers.loc[:origin].to_numpy()
            # apply() reuses the frozen parameters on the longer history.
            arima_path = arima_fit.apply(history_y).forecast(max(STRESS_HORIZONS))
            future_x = np.repeat(history_x[-1:], max(STRESS_HORIZONS), axis=0)
            sarimax_path = sarimax_fit.apply(history_y, exog=history_x).forecast(
                max(STRESS_HORIZONS), exog=future_x
            )
            for horizon in STRESS_HORIZONS:
                target = origin + horizon
                if target >= len(lane) or pd.isna(lane.loc[target, "y"]):
                    continue
                lightgbm_change = frozen_lightgbm[horizon].predict(
                    lane.loc[[origin], LIGHTGBM_FEATURES]
                )[0]
                forecasts = {
                    "naive": history_y[-1],
                    "arima": arima_path[horizon - 1],
                    "sarimax": sarimax_path[horizon - 1],
                    "lightgbm": history_y[-1] + lightgbm_change,
                }
                for model_name, forecast in forecasts.items():
                    frozen_rows.append(
                        {
                            "shock": shock,
                            "mode": "frozen",
                            "model": model_name,
                            "lane_id": lane_id,
                            "origin_date": lane.loc[origin, "week_date"],
                            "target_date": lane.loc[target, "week_date"],
                            "horizon": horizon,
                            "error_pct": 100
                            * abs(np.exp(forecast - lane.loc[target, "y"]) - 1),
                        }
                    )
frozen = pd.DataFrame(frozen_rows)

# ---------------------------------------------------------------
# 2. Retrained forecasts for the same origins (from the backtests)
# ---------------------------------------------------------------
backtest = pd.concat(
    [
        pd.read_csv(processed_folder / "forecasts_baselines.csv"),
        pd.read_csv(processed_folder / "forecasts_models.csv"),
    ],
    ignore_index=True,
)
backtest["origin_date"] = pd.to_datetime(backtest["origin_date"])
backtest["target_date"] = pd.to_datetime(backtest["target_date"])
retrained_parts = []
for shock, freeze_text in SHOCKS.items():
    in_window = (
        (backtest["origin_date"] >= pd.Timestamp(freeze_text))
        & backtest["model"].isin(["naive", "arima", "sarimax", "lightgbm"])
        & backtest["horizon"].isin(STRESS_HORIZONS)
    )
    part = backtest[in_window].copy()
    part["shock"] = shock
    part["mode"] = "retrained"
    part["error_pct"] = 100 * (np.exp(part["y_forecast"] - part["y_actual"]) - 1).abs()
    retrained_parts.append(part[frozen.columns])
errors = pd.concat([frozen] + retrained_parts, ignore_index=True)
# The Red Sea window is cut where the Hormuz shock starts.
red_sea_end = pd.Timestamp(SHOCKS["hormuz"])
errors = errors[
    ~((errors["shock"] == "red_sea") & (errors["target_date"] > red_sea_end))
]
errors.to_csv(reports_folder / "stress_test_errors.csv", index=False)

# ---------------------------------------------------------------
# 3. Summary: mean error by shock, lane, model, horizon and mode
# ---------------------------------------------------------------
summary = (
    errors.groupby(["shock", "lane_id", "model", "horizon", "mode"])["error_pct"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count": "n_forecasts", "mean": "mean_error_pct"})
)
summary["mean_error_pct"] = summary["mean_error_pct"].round(2)
summary.to_csv(reports_folder / "stress_test_summary.csv", index=False)

main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
for shock in SHOCKS:
    view = summary[(summary["shock"] == shock) & (summary["lane_id"] == main_lane)]
    print(f"{shock}, {main_lane}: mean error (% of rate)")
    print(
        view.pivot_table(
            index="model", columns=["horizon", "mode"], values="mean_error_pct"
        ).to_string()
    )

# ---------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------
# Error trajectory on the main lane, 4 weeks ahead, frozen vs retrained.

for shock in SHOCKS:
    figure, axis = plt.subplots(figsize=(10, 4))
    view = errors[
        (errors["shock"] == shock)
        & (errors["lane_id"] == main_lane)
        & (errors["horizon"] == 4)
    ]
    for model_name in CHART_MODELS:
        color = MODEL_COLORS[model_name]
        for mode, style in [("frozen", ":"), ("retrained", "-")]:
            line = view[(view["model"] == model_name) & (view["mode"] == mode)]
            if model_name == "naive" and mode == "frozen":
                continue  # the naive forecast has no parameters to freeze
            line = line.sort_values("target_date")
            axis.plot(
                line["target_date"],
                line["error_pct"],
                linestyle=style,
                color=color,
                label=f"{model_name} ({mode})" if model_name != "naive" else "naive",
            )
    axis.set_ylabel("Absolute error, % of rate")
    shock_name = SHOCK_NAMES[shock]
    axis.set_title(f"{LANE_LABELS[main_lane]}: 4-week-ahead errors, {shock_name} shock")
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(
        figures_folder / f"stress_errors_{shock}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(figure)

# Exposure gradient in the errors: the naive error measures how much
# each lane moved in 4 weeks during the Hormuz period.
view = summary[
    (summary["shock"] == "hormuz")
    & (summary["horizon"] == 4)
    & (summary["mode"] == "retrained")
]
figure, axis = plt.subplots(figsize=(8, 3.6))
for position, lane_id in enumerate(LANE_COLORS):
    lane_view = view[view["lane_id"] == lane_id].set_index("model")["mean_error_pct"]
    axis.bar(
        position, lane_view.get("naive", np.nan), width=0.6, color=LANE_COLORS[lane_id]
    )
    axis.text(
        position,
        lane_view.get("naive", 0),
        f"{lane_view.get('naive', 0):.0f}%",
        ha="center",
        va="bottom",
        fontsize=8,
    )
axis.set_xticks(range(len(LANE_COLORS)), [LANE_LABELS[lane] for lane in LANE_COLORS])
axis.set_ylabel("Mean error, % of rate")
axis.set_title("Naive 4-week-ahead error during the Hormuz period, by lane")
axis.axhline(0, color=TEXT_MUTED, linewidth=0.6)
figure.savefig(figures_folder / "stress_lane_errors.png", dpi=150, bbox_inches="tight")
plt.close(figure)
print("Saved stress test tables and figures")

# ---------------------------------------------------------------
# 5. MLflow: one run with every mean error and the figures
# ---------------------------------------------------------------
# Metric names: error_<shock>_<lane>_<model>_<mode>_h<h>, the mean
# error in percent of the actual rate.
start_tracking("stress_test")
with mlflow.start_run(run_name="hormuz_and_red_sea"):
    mlflow.log_params({"freeze_dates": SHOCKS, "horizons": STRESS_HORIZONS})
    stress_metrics = {}
    for row in summary.itertuples():
        name = f"error_{row.shock}_{row.lane_id}_{row.model}_{row.mode}_h{row.horizon}"
        stress_metrics[name] = row.mean_error_pct
    log_finite_metrics(stress_metrics)
    mlflow.log_artifact(str(reports_folder / "stress_test_summary.csv"))
    for figure_path in sorted(figures_folder.glob("stress_*.png")):
        mlflow.log_artifact(str(figure_path))
print("Logged the run to MLflow (experiment: stress_test)")
