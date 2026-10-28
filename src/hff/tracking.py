# ---------------------------------------------------------------
# Module: hff/tracking.py
# Purpose: Records every pipeline run in MLflow, so that models and
#          runs can be compared side by side in the MLflow interface
#          ("make mlflow", then http://127.0.0.1:5000).
#
# Store: a local SQLite database (mlflow.db) for parameters and
# metrics, and a folder (mlartifacts/) for files such as tables and
# figures. Both are git-ignored and rebuilt by the scripts.
#
# Structure in MLflow:
#   experiment   one per pipeline step: "backtest", "stress_test",
#                "business" (scenarios, pricing), "weekly_forecast"
#   parent run   one execution of a script, with the shared settings
#                and the result tables as artifacts
#   child run    one per model, with its own parameters and metrics
#
# Metric names: <measure>_<lane>_h<h>, for example
#   mase_asia_middle_east_h4   MASE on the Middle East lane, 4 weeks
#   coverage_asia_middle_east_h4, pinball_asia_middle_east_h4
# plus, for the main lane, mase_vs_naive_h<h> = MASE of the model
# divided by MASE of the naive forecast (below 1 = better than
# "no change"), the quickest column to compare in the interface.
# Every metric is over the whole backtest (period "all").
# ---------------------------------------------------------------

import math

import mlflow
import pandas as pd

from hff.config import PROJECT_ROOT

TRACKING_DATABASE = PROJECT_ROOT / "mlflow.db"
ARTIFACT_FOLDER = PROJECT_ROOT / "mlartifacts"
TRACKING_URI = f"sqlite:///{TRACKING_DATABASE.as_posix()}"


def start_tracking(experiment_name: str) -> None:
    """Points MLflow at the project's local store and selects an experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_location = (ARTIFACT_FOLDER / experiment_name).as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def backtest_settings(config: dict) -> dict:
    """Settings shared by every backtest run, logged as parameters."""
    return {
        "start_date": config["start_date"],
        "horizons": config["horizons"],
        "warm_up_weeks": config["backtest_warm_up_weeks"],
        "arima_order": config["arima_order"],
        "retrain_every_weeks": config["retrain_every_weeks"],
        "random_seed": config["random_seed"],
    }


def log_finite_metrics(metrics: dict) -> None:
    """Logs metrics, leaving out missing values (NaN), which MLflow
    cannot sort or compare."""
    finite = {}
    for name, value in metrics.items():
        if value is not None and not math.isnan(float(value)):
            finite[name] = float(value)
    if finite:
        mlflow.log_metrics(finite)


def point_metrics_for(
    leaderboard: pd.DataFrame, model_name: str, main_lane: str
) -> dict:
    """MASE, MAPE and directional accuracy of one model on every lane
    and horizon, plus its MASE relative to the naive forecast on the
    main lane."""
    overall = leaderboard[leaderboard["period"] == "all"]
    rows = overall[overall["model"] == model_name]
    metrics = {}
    for row in rows.itertuples():
        suffix = f"{row.lane_id}_h{row.horizon}"
        metrics[f"mase_{suffix}"] = row.mase
        metrics[f"mape_{suffix}"] = row.mape_pct
        metrics[f"directional_{suffix}"] = row.directional_accuracy

    naive = overall[(overall["model"] == "naive") & (overall["lane_id"] == main_lane)]
    naive_mase = naive.set_index("horizon")["mase"]
    for row in rows[rows["lane_id"] == main_lane].itertuples():
        if row.horizon in naive_mase.index:
            metrics[f"mase_vs_naive_h{row.horizon}"] = (
                row.mase / naive_mase[row.horizon]
            )
    return metrics


def log_backtest_run(
    run_name: str,
    config: dict,
    child_settings: dict,
    leaderboard: pd.DataFrame,
    artifact_paths: list,
    intervals: pd.DataFrame | None = None,
    child_extra_metrics: dict | None = None,
) -> None:
    """Logs one backtest script as a parent run with one child run per
    model.

    child_settings maps each model name to its parameters. The parent
    run holds the shared settings and the result tables; each child run
    holds the model's point metrics and, when intervals is given, its
    interval metrics. child_extra_metrics adds metrics per model (for
    example the number of failed fits).
    """
    main_lane = [lane["id"] for lane in config["lanes"] if lane["role"] == "main"][0]
    start_tracking("backtest")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(backtest_settings(config))
        for path in artifact_paths:
            mlflow.log_artifact(str(path))
        for model_name, settings in child_settings.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.set_tag("model", model_name)
                mlflow.log_params(settings)
                log_finite_metrics(
                    point_metrics_for(leaderboard, model_name, main_lane)
                )
                if intervals is not None:
                    log_finite_metrics(interval_metrics_for(intervals, model_name))
                if child_extra_metrics is not None:
                    log_finite_metrics(child_extra_metrics.get(model_name, {}))
    print(f"Logged the run '{run_name}' to MLflow (experiment: backtest)")


def interval_metrics_for(intervals: pd.DataFrame, model_name: str) -> dict:
    """Coverage and pinball loss of one model's 80% interval on every
    lane and horizon."""
    overall = intervals[intervals["period"] == "all"]
    rows = overall[overall["model"] == model_name]
    metrics = {}
    for row in rows.itertuples():
        suffix = f"{row.lane_id}_h{row.horizon}"
        metrics[f"coverage_{suffix}"] = row.coverage_80
        metrics[f"pinball_{suffix}"] = row.pinball_loss
    return metrics
