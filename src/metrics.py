import numpy as np
from sklearn.metrics import r2_score

from src.splits import PHASES, phase_mask


if hasattr(np, "trapz"):
    _trapz = np.trapz
else:
    _trapz = np.trapezoid


def curve_metrics(loss: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - loss
    rel = np.abs(error) / np.maximum(np.abs(loss), 1e-12)
    auc_loss = _trapz(loss)
    auc_pred = _trapz(pred)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "prede": float(np.mean(rel)),
        "worste": float(np.max(rel)),
        "r2": float(r2_score(loss, pred)),
        "final_rel_error": float(abs(pred[-1] - loss[-1]) / max(abs(loss[-1]), 1e-12)),
        "auc_rel_error": float(abs(auc_pred - auc_loss) / max(abs(auc_loss), 1e-12)),
    }


def stage_metrics(steps: np.ndarray, loss: np.ndarray, pred: np.ndarray) -> list[dict]:
    rows = []
    for phase_name, (start, end) in PHASES.items():
        mask = phase_mask(steps, phase_name)
        n_points = int(mask.sum())
        if n_points == 0:
            rows.append(
                {
                    "stage": phase_name,
                    "start_step": start,
                    "end_step": end,
                    "n_points": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "prede": np.nan,
                    "mean_signed_error": np.nan,
                }
            )
            continue
        error = pred[mask] - loss[mask]
        rel = np.abs(error) / np.maximum(np.abs(loss[mask]), 1e-12)
        rows.append(
            {
                "stage": phase_name,
                "start_step": start,
                "end_step": end,
                "n_points": n_points,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "prede": float(np.mean(rel)),
                "mean_signed_error": float(np.mean(error)),
            }
        )
    return rows
