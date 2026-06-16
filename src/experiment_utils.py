import csv
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import huber

from src.config import FOLDER_PATHS, PARAMS
from src.data_loader import load_data
from src.metrics import curve_metrics, stage_metrics
from src.predictors import predict_mpl_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_results_dirs():
    for path in [METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_project_cwd():
    os.chdir(PROJECT_ROOT)


def load_size_data(size: str) -> dict:
    set_project_cwd()
    return load_data(FOLDER_PATHS[size])


def mpl_multistart_initializations(size: str) -> list[np.ndarray]:
    base = np.array(PARAMS[size], dtype=float)
    variants = [base.copy()]
    for scale in [0.95, 1.05, 0.9, 1.1]:
        candidate = base.copy()
        positive = candidate > 0.0
        candidate[positive] *= scale
        variants.append(candidate)
    return variants


def summarize_numeric_rows(rows: list[dict], group_keys: list[str], value_keys: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(str(row.get(group_key, "")) for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    summary = []
    for key in sorted(groups):
        output = dict(zip(group_keys, key))
        group_rows = groups[key]
        for value_key in value_keys:
            values = []
            for row in group_rows:
                try:
                    value = float(row.get(value_key, ""))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)

            if values:
                array = np.array(values, dtype=float)
                output[f"{value_key}_mean"] = float(np.mean(array))
                output[f"{value_key}_std"] = float(np.std(array, ddof=0))
                output[f"{value_key}_min"] = float(np.min(array))
                output[f"{value_key}_max"] = float(np.max(array))
            else:
                output[f"{value_key}_mean"] = float("nan")
                output[f"{value_key}_std"] = float("nan")
                output[f"{value_key}_min"] = float("nan")
                output[f"{value_key}_max"] = float("nan")
        summary.append(output)
    return summary


def fit_mpl_scipy(data: dict, train_curves: list[str], size: str, maxiter: int = 300) -> tuple[list[float], float]:
    init = np.array(PARAMS[size], dtype=float)

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


def prediction_rows(size: str, method: str, curve: str, steps, loss, pred) -> list[dict]:
    return [
        {
            "step": int(step),
            "loss": float(actual),
            "pred": float(estimated),
            "residual": float(actual - estimated),
            "method": method,
            "size": size,
            "curve": curve,
        }
        for step, actual, estimated in zip(steps, loss, pred)
    ]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_csv(run_id: str, size: str, method: str, curve: str, rows: list[dict]):
    curve_id = curve.replace(".csv", "")
    path = PREDICTIONS_DIR / f"{run_id}_{size}_{method}_{curve_id}.csv"
    write_csv(path, rows, ["step", "loss", "pred", "residual", "method", "size", "curve"])


def metric_row(base: dict, loss: np.ndarray, pred: np.ndarray) -> dict:
    row = dict(base)
    row.update(curve_metrics(loss, pred))
    return row


def stage_metric_rows(base: dict, steps: np.ndarray, loss: np.ndarray, pred: np.ndarray) -> list[dict]:
    rows = []
    for stage_row in stage_metrics(steps, loss, pred):
        row = dict(base)
        row.update(stage_row)
        rows.append(row)
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_feature_columns(features: np.ndarray, names: list[str], selected_names: list[str]) -> tuple[np.ndarray, list[str]]:
    indices = [names.index(name) for name in selected_names]
    return features[:, indices], selected_names
