import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annealing_law import fit_best_tissue
from src.correction import RidgeResidualCorrection, stack_residual_training_rows
from src.experiment_utils import (
    ensure_results_dirs,
    fit_mpl_scipy,
    load_size_data,
    metric_row,
    select_feature_columns,
    write_csv,
)
from src.features import correction_features, fsl_light_features
from src.predictors import predict_mpl_curve
from src.splits import MAIN_TEST_CURVES, SIZES, STRICT_COSINE_TRAIN_CURVES


RUN_ID = "ablation_cosine_train"
MAIN_ALPHA = 1e-4
ALPHA_SWEEP = [1e-6, 1e-4, 1e-2, 1.0]

FEATURE_SETS = {
    "bias_only": ["bias"],
    "lr_level": ["bias", "eta_norm", "t_norm"],
    "cumulative": ["bias", "log_s1", "s1_sq", "t_norm"],
    "decay_cum": ["bias", "log_s1", "eta_norm", "decay_cum", "t_norm"],
    "decay_memory": [
        "bias",
        "log_s1",
        "eta_norm",
        "decay_mem_099",
        "decay_mem_0995",
        "decay_mem_0999",
        "t_norm",
    ],
    "full": [
        "bias",
        "log_s1",
        "eta_norm",
        "s1_sq",
        "decay_cum",
        "decay_mem_099",
        "decay_mem_0995",
        "decay_mem_0999",
        "t_norm",
        "is_decay",
    ],
    "fsl_light": "fsl_light",
}


def make_feature_fn(selected_names):
    if selected_names == "fsl_light":
        return fsl_light_features

    def feature_fn(lrs, steps):
        features, names = correction_features(lrs, steps)
        return select_feature_columns(features, names, selected_names)

    return feature_fn


def fit_residual_correction(data, base_predictions, feature_fn, alpha):
    features, residual, names = stack_residual_training_rows(
        data, STRICT_COSINE_TRAIN_CURVES, base_predictions, feature_fn
    )
    return RidgeResidualCorrection(alpha=alpha).fit(features, residual, names)


def evaluate_correction(data, predict, correction, feature_fn, size, base_method, feature_set, alpha=None):
    rows = []
    for curve in MAIN_TEST_CURVES:
        base_pred = predict(curve)
        eval_features, _ = feature_fn(data[curve]["lrs"], data[curve]["step"])
        pred = base_pred + correction.predict_residual(eval_features)
        base = {
            "run_id": RUN_ID,
            "size": size,
            "base_method": base_method,
            "feature_set": feature_set,
            "test_curve": curve,
        }
        if alpha is not None:
            base["alpha"] = alpha
        rows.append(metric_row(base, data[curve]["loss"], pred))
    return rows


def main():
    ensure_results_dirs()
    metric_rows = []
    alpha_sweep_rows = []
    coefficient_rows = []

    for size in SIZES:
        data = load_size_data(size)
        tissue = fit_best_tissue(data, STRICT_COSINE_TRAIN_CURVES)
        mpl_params, _ = fit_mpl_scipy(data, STRICT_COSINE_TRAIN_CURVES, size)

        base_predictors = {
            "tissue": lambda curve: tissue["model"].predict_curve(data[curve]["lrs"], data[curve]["step"]),
            "mpl": lambda curve: predict_mpl_curve(data[curve]["lrs"], data[curve]["step"], mpl_params),
        }

        for base_method, predict in base_predictors.items():
            base_predictions = {curve: predict(curve) for curve in STRICT_COSINE_TRAIN_CURVES}
            for feature_set, selected_names in FEATURE_SETS.items():
                feature_fn = make_feature_fn(selected_names)
                correction = fit_residual_correction(data, base_predictions, feature_fn, MAIN_ALPHA)

                if feature_set == "full":
                    for name, coefficient in zip(correction.feature_names_, correction.coef_):
                        coefficient_rows.append(
                            {
                                "run_id": RUN_ID,
                                "size": size,
                                "base_method": base_method,
                                "feature_name": name,
                                "coefficient": float(coefficient),
                            }
                        )

                metric_rows.extend(
                    evaluate_correction(data, predict, correction, feature_fn, size, base_method, feature_set)
                )

            fsl_feature_fn = make_feature_fn(FEATURE_SETS["fsl_light"])
            for alpha in ALPHA_SWEEP:
                correction = fit_residual_correction(data, base_predictions, fsl_feature_fn, alpha)
                alpha_sweep_rows.extend(
                    evaluate_correction(
                        data,
                        predict,
                        correction,
                        fsl_feature_fn,
                        size,
                        base_method,
                        "fsl_light",
                        alpha=alpha,
                    )
                )

    write_csv(
        ROOT / "results" / "metrics" / "ablation_metrics.csv",
        metric_rows,
        [
            "run_id",
            "size",
            "base_method",
            "feature_set",
            "test_curve",
            "rmse",
            "mae",
            "prede",
            "worste",
            "r2",
            "final_rel_error",
            "auc_rel_error",
        ],
    )
    write_csv(
        ROOT / "results" / "metrics" / "correction_coefficients.csv",
        coefficient_rows,
        ["run_id", "size", "base_method", "feature_name", "coefficient"],
    )
    write_csv(
        ROOT / "results" / "metrics" / "alpha_sweep_metrics.csv",
        alpha_sweep_rows,
        [
            "run_id",
            "size",
            "base_method",
            "feature_set",
            "alpha",
            "test_curve",
            "rmse",
            "mae",
            "prede",
            "worste",
            "r2",
            "final_rel_error",
            "auc_rel_error",
        ],
    )


if __name__ == "__main__":
    main()
