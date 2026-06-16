import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import huber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_utils import (
    METRICS_DIR,
    ensure_results_dirs,
    load_size_data,
    metric_row,
    mpl_multistart_initializations,
    read_csv_rows,
    summarize_numeric_rows,
    write_csv,
)
from src.predictors import predict_mpl_curve
from src.splits import MAIN_TEST_CURVES, SIZES, STRICT_COSINE_TRAIN_CURVES


RUN_ID = "direction_b_robustness"
MULTISTART_VALUE_KEYS = ["rmse", "final_rel_error", "auc_rel_error", "objective"]
MULTISTART_METRIC_FIELDS = [
    "run_id",
    "size",
    "start_id",
    "test_curve",
    "objective",
    "rmse",
    "mae",
    "prede",
    "worste",
    "r2",
    "final_rel_error",
    "auc_rel_error",
]
MULTISTART_SUMMARY_FIELDS = ["size", "test_curve"] + [
    f"{value_key}_{stat}"
    for value_key in MULTISTART_VALUE_KEYS
    for stat in ["mean", "std", "min", "max"]
]
STAGE_SENSITIVITY_FIELDS = [
    "run_id",
    "size",
    "method",
    "test_curve",
    "stable_rmse",
    "decay_rmse",
    "decay_minus_stable_rmse",
    "decay_to_stable_rmse_ratio",
]


def fit_mpl_from_initialization(data: dict, train_curves: list[str], init, maxiter: int = 300) -> tuple[list[float], float]:
    init = np.array(init, dtype=float)

    def objective(params):
        if np.any(params <= 0):
            return 1e9
        total = 0.0
        for name in train_curves:
            pred = predict_mpl_curve(data[name]["lrs"], data[name]["step"], params)
            pred = np.maximum(pred, 1e-10)
            residual = np.log(data[name]["loss"]) - np.log(pred)
            total += float(huber(0.001, residual).sum())
        return total

    result = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=[
            (1e-8, None),
            (1e-8, None),
            (1e-8, 5.0),
            (1e-8, None),
            (1e-8, None),
            (1e-8, 5.0),
            (1e-8, 5.0),
        ],
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    return result.x.tolist(), float(result.fun)


def multistart_metric_rows(size: str, data: dict, maxiter: int = 300) -> list[dict]:
    rows = []
    for start_id, init in enumerate(mpl_multistart_initializations(size)):
        params, objective = fit_mpl_from_initialization(data, STRICT_COSINE_TRAIN_CURVES, init, maxiter=maxiter)
        for curve in MAIN_TEST_CURVES:
            pred = predict_mpl_curve(data[curve]["lrs"], data[curve]["step"], params)
            rows.append(
                metric_row(
                    {
                        "run_id": RUN_ID,
                        "size": size,
                        "start_id": start_id,
                        "test_curve": curve,
                        "objective": objective,
                    },
                    data[curve]["loss"],
                    pred,
                )
            )
    return rows


def multistart_summary_rows(rows: list[dict]) -> list[dict]:
    return summarize_numeric_rows(rows, ["size", "test_curve"], MULTISTART_VALUE_KEYS)


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def stage_sensitivity_rows(stage_rows: list[dict]) -> list[dict]:
    paired = {}
    stages = {"post_warmup_stable", "decay"}
    for row in stage_rows:
        if row.get("test_curve") not in MAIN_TEST_CURVES or row.get("stage") not in stages:
            continue
        key = (
            row.get("run_id", ""),
            row.get("size", ""),
            row.get("method", ""),
            row.get("test_curve", ""),
        )
        paired.setdefault(key, {})[row["stage"]] = row

    rows = []
    for key in sorted(paired):
        group = paired[key]
        if "post_warmup_stable" not in group or "decay" not in group:
            continue
        stable_rmse = _float_or_nan(group["post_warmup_stable"].get("rmse"))
        decay_rmse = _float_or_nan(group["decay"].get("rmse"))
        ratio = decay_rmse / stable_rmse if stable_rmse != 0.0 else float("nan")
        rows.append(
            {
                "run_id": key[0],
                "size": key[1],
                "method": key[2],
                "test_curve": key[3],
                "stable_rmse": stable_rmse,
                "decay_rmse": decay_rmse,
                "decay_minus_stable_rmse": decay_rmse - stable_rmse,
                "decay_to_stable_rmse_ratio": ratio,
            }
        )
    return rows


def main():
    ensure_results_dirs()

    stage_metrics_path = METRICS_DIR / "stage_metrics.csv"
    if not stage_metrics_path.exists():
        raise FileNotFoundError(
            f"Required stage metrics file is missing: {stage_metrics_path}. "
            "Run `python scripts/run_cross_schedule.py` first."
        )

    metric_rows = []
    for size in SIZES:
        metric_rows.extend(multistart_metric_rows(size, load_size_data(size)))

    write_csv(METRICS_DIR / "mpl_multistart_metrics.csv", metric_rows, MULTISTART_METRIC_FIELDS)
    write_csv(
        METRICS_DIR / "mpl_multistart_summary.csv",
        multistart_summary_rows(metric_rows),
        MULTISTART_SUMMARY_FIELDS,
    )

    stage_rows = read_csv_rows(stage_metrics_path)
    write_csv(
        METRICS_DIR / "stage_sensitivity_metrics.csv",
        stage_sensitivity_rows(stage_rows),
        STAGE_SENSITIVITY_FIELDS,
    )


if __name__ == "__main__":
    main()
