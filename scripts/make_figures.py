import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_utils import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, ensure_results_dirs, load_size_data
from src.splits import MAIN_TEST_CURVES, SIZES


CROSS_RUN_ID = "cross_schedule_cosine_train"
ABLATION_RUN_ID = "ablation_cosine_train"
SCHEDULE_CURVES = [
    "cosine_24000.csv",
    "constant_24000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_18.csv",
]
PLOT_METHODS = [
    "tissue",
    "tissue_plus_ridge",
    "tissue_plus_fsl_light",
    "mpl",
    "mpl_plus_ridge",
    "mpl_plus_fsl_light",
]
MAIN_COMPARISON_METHODS = ["tissue", "mpl", "mpl_plus_ridge"]
FSL_COMPARISON_METHODS = [
    "tissue",
    "tissue_plus_ridge",
    "tissue_plus_fsl_light",
    "mpl",
    "mpl_plus_ridge",
    "mpl_plus_fsl_light",
]
NCPL_METHODS = ["ncpl_surrogate"]
STAGE_PLOT_STAGES = ["post_warmup_stable", "decay"]


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_prediction(size, method, curve):
    curve_id = curve.replace(".csv", "")
    path = PREDICTIONS_DIR / f"{CROSS_RUN_ID}_{size}_{method}_{curve_id}.csv"
    rows = read_rows(path)
    if not rows:
        return None
    return {
        "step": np.array([int(row["step"]) for row in rows]),
        "loss": np.array([float(row["loss"]) for row in rows]),
        "pred": np.array([float(row["pred"]) for row in rows]),
        "residual": np.array([float(row["residual"]) for row in rows]),
    }


def plot_lr_schedules():
    for size in SIZES:
        data = load_size_data(size)
        plt.figure(figsize=(8, 4))
        for curve in SCHEDULE_CURVES:
            plt.plot(data[curve]["lrs"], label=curve.replace(".csv", ""))
        plt.xlabel("Step")
        plt.ylabel("Learning rate")
        plt.title(f"Learning-rate schedules ({size}M)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"lr_schedules_{size}.png", dpi=160)
        plt.close()


def plot_prediction_diagnostics():
    for size in SIZES:
        for curve in MAIN_TEST_CURVES:
            loaded = {method: read_prediction(size, method, curve) for method in PLOT_METHODS}
            loaded = {method: values for method, values in loaded.items() if values is not None}
            if not loaded:
                continue
            first = next(iter(loaded.values()))
            curve_id = curve.replace(".csv", "")

            plt.figure(figsize=(8, 4))
            plt.plot(first["step"], first["loss"], label="true", linewidth=2)
            for method, values in loaded.items():
                plt.plot(values["step"], values["pred"], label=method, linestyle="--")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"{size}M {curve_id}: prediction")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"prediction_{size}_{curve_id}.png", dpi=160)
            plt.close()

            plt.figure(figsize=(8, 4))
            for method, values in loaded.items():
                plt.plot(values["step"], values["residual"], label=method)
            plt.axhline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Step")
            plt.ylabel("True - predicted")
            plt.title(f"{size}M {curve_id}: residual")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"residual_{size}_{curve_id}.png", dpi=160)
            plt.close()


def plot_main_comparison():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "cross_schedule_metrics.csv")
        if row["test_curve"] in MAIN_TEST_CURVES and row["method"] in MAIN_COMPARISON_METHODS
    ]
    if not rows:
        return
    labels = [f"{row['size']}M\n{row['method']}\n{row['test_curve'].split('_')[0]}" for row in rows]
    values = [float(row["rmse"]) for row in rows]
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("RMSE")
    plt.title("Cosine-trained prediction error on WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "main_comparison_grid.png", dpi=160)
    plt.close()


def plot_final_error():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "cross_schedule_metrics.csv")
        if row["test_curve"] in MAIN_TEST_CURVES and row["method"] in MAIN_COMPARISON_METHODS
    ]
    if not rows:
        return
    labels = [f"{row['size']}M\n{row['method']}\n{row['test_curve'].split('_')[0]}" for row in rows]
    values = [float(row["final_rel_error"]) for row in rows]
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("Final relative error")
    plt.title("Final-loss prediction error on WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_error_bar.png", dpi=160)
    plt.close()


def method_label(method):
    labels = {
        "tissue": "Tissue",
        "tissue_plus_ridge": "Tissue + Ridge",
        "tissue_plus_fsl_light": "Tissue + FSL-light",
        "mpl": "MPL",
        "mpl_plus_ridge": "MPL + Ridge",
        "mpl_plus_fsl_light": "MPL + FSL-light",
        "ncpl_surrogate": "NCPL-style surrogate",
    }
    return labels.get(method, method)


def filter_stage_rows(rows, methods):
    return [
        row
        for row in rows
        if row["test_curve"] in MAIN_TEST_CURVES
        and row["stage"] in STAGE_PLOT_STAGES
        and row["method"] in methods
    ]


def summarize_stage_errors(rows):
    grouped = {}
    for row in rows:
        rmse = float(row["rmse"])
        if not np.isfinite(rmse):
            continue
        key = (row["size"], row["method"], row["stage"])
        grouped.setdefault(key, []).append(rmse)

    summary = []
    for size in [str(size) for size in SIZES]:
        for method in PLOT_METHODS:
            for stage in STAGE_PLOT_STAGES:
                values = grouped.get((size, method, stage))
                if not values:
                    continue
                summary.append(
                    {
                        "size": size,
                        "method": method,
                        "stage": stage,
                        "rmse": float(np.mean(values)),
                    }
                )
    return summary


def plot_fsl_light_comparison():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "cross_schedule_metrics.csv")
        if row["test_curve"] in MAIN_TEST_CURVES and row["method"] in FSL_COMPARISON_METHODS
    ]
    if not rows:
        return

    labels = [
        f"{row['size']}M\n{method_label(row['method'])}\n{row['test_curve'].split('_')[0].upper()}"
        for row in rows
    ]
    values = [float(row["rmse"]) for row in rows]
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("RMSE")
    plt.title("Baseline vs full ridge vs FSL-light on WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fsl_light_comparison.png", dpi=160)
    plt.close()


def plot_stage_errors():
    rows = filter_stage_rows(read_rows(METRICS_DIR / "stage_metrics.csv"), MAIN_COMPARISON_METHODS)
    summary = summarize_stage_errors(rows)
    if not summary:
        return
    labels = [f"{row['size']}M\n{row['method']}\n{row['stage']}" for row in summary]
    values = [row["rmse"] for row in summary]
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("Stage RMSE")
    plt.title("Stage-wise prediction error averaged over WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage_error_bar.png", dpi=160)
    plt.close()


def plot_stage_fsl_light_error():
    rows = filter_stage_rows(read_rows(METRICS_DIR / "stage_metrics.csv"), FSL_COMPARISON_METHODS)
    summary = summarize_stage_errors(rows)
    if not summary:
        return

    labels = [
        f"{row['size']}M\n{method_label(row['method'])}\n{row['stage'].replace('post_warmup_', '')}"
        for row in summary
    ]
    values = [row["rmse"] for row in summary]
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("Stage RMSE")
    plt.title("Stage-wise FSL-light comparison averaged over WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage_fsl_light_error.png", dpi=160)
    plt.close()


def plot_alpha_sweep_sensitivity():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "alpha_sweep_metrics.csv")
        if row["test_curve"] in MAIN_TEST_CURVES and row["base_method"] in ["tissue", "mpl"]
    ]
    if not rows:
        return

    grouped = {}
    for row in rows:
        key = (row["size"], row["base_method"], row["test_curve"])
        grouped.setdefault(key, []).append((float(row["alpha"]), float(row["rmse"])))

    plt.figure(figsize=(10, 5))
    for size in [str(size) for size in SIZES]:
        for base_method in ["tissue", "mpl"]:
            for test_curve in MAIN_TEST_CURVES:
                values = grouped.get((size, base_method, test_curve))
                if not values:
                    continue
                values = sorted(values)
                alphas = [value[0] for value in values]
                rmses = [value[1] for value in values]
                curve_label = test_curve.split("_")[0].upper()
                plt.plot(alphas, rmses, marker="o", label=f"{size}M {method_label(base_method)} {curve_label}")

    plt.xscale("log")
    plt.xlabel("Ridge alpha")
    plt.ylabel("RMSE")
    plt.title("FSL-light alpha sensitivity on WSD/WSDLD")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "alpha_sweep_sensitivity.png", dpi=160)
    plt.close()


def plot_mpl_multistart_uncertainty():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "mpl_multistart_summary.csv")
        if row["test_curve"] in MAIN_TEST_CURVES
    ]
    if not rows:
        return

    labels = [f"{row['size']}M\n{row['test_curve'].split('_')[0].upper()}" for row in rows]
    values = [float(row["rmse_mean"]) for row in rows]
    errors = [float(row["rmse_std"]) for row in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(values)), values, yerr=errors, capsize=4)
    plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("RMSE mean +/- std")
    plt.title("MPL multi-start prediction uncertainty")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mpl_multistart_uncertainty.png", dpi=160)
    plt.close()


def plot_stage_sensitivity():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "stage_sensitivity_metrics.csv")
        if row["test_curve"] in MAIN_TEST_CURVES and row["method"] in FSL_COMPARISON_METHODS
    ]
    if not rows:
        return

    labels = [
        f"{row['size']}M\n{method_label(row['method'])}\n{row['test_curve'].split('_')[0].upper()}"
        for row in rows
    ]
    values = [float(row["decay_minus_stable_rmse"]) for row in rows]
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(values)), values)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("Decay RMSE - stable RMSE")
    plt.title("Stage sensitivity on WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage_sensitivity.png", dpi=160)
    plt.close()


def plot_ncpl_surrogate_comparison():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "ncpl_surrogate_metrics.csv")
        if row["test_curve"] in MAIN_TEST_CURVES and row["method"] in NCPL_METHODS
    ]
    if not rows:
        return

    labels = [
        f"{row['size']}M\n{method_label(row['method'])}\n{row['test_curve'].split('_')[0].upper()}"
        for row in rows
    ]
    values = [float(row["rmse"]) for row in rows]
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("RMSE")
    plt.title("Direction C: NCPL-style direct surrogate on WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ncpl_surrogate_comparison.png", dpi=160)
    plt.close()


def plot_ablation():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "ablation_metrics.csv")
        if row["base_method"] == "mpl" and row["test_curve"] == "wsd_20000_24000.csv"
    ]
    if not rows:
        return
    labels = [f"{row['size']}M\n{row['feature_set']}" for row in rows]
    values = [float(row["rmse"]) for row in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("RMSE")
    plt.title("Residual correction feature ablation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_bar.png", dpi=160)
    plt.close()


def plot_coefficients():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "correction_coefficients.csv")
        if row["size"] == "400" and row["base_method"] == "mpl"
    ]
    if not rows:
        return
    labels = [row["feature_name"] for row in rows]
    values = [float(row["coefficient"]) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("Coefficient")
    plt.title("MPL full-feature ridge coefficients (400M)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correction_coefficients.png", dpi=160)
    plt.close()


def main():
    ensure_results_dirs()
    plot_lr_schedules()
    plot_prediction_diagnostics()
    plot_main_comparison()
    plot_final_error()
    plot_fsl_light_comparison()
    plot_stage_errors()
    plot_stage_fsl_light_error()
    plot_ablation()
    plot_coefficients()
    plot_alpha_sweep_sensitivity()
    plot_mpl_multistart_uncertainty()
    plot_stage_sensitivity()
    plot_ncpl_surrogate_comparison()


if __name__ == "__main__":
    main()
