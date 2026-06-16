import sys
from dataclasses import dataclass
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
    prediction_rows,
    stage_metric_rows,
    write_csv,
    write_prediction_csv,
)
from src.features import correction_features, fsl_light_features
from src.predictors import predict_mpl_curve
from src.splits import ALL_EVAL_CURVES, SIZES, STRICT_COSINE_TRAIN_CURVES


RUN_ID = "cross_schedule_cosine_train"


@dataclass(frozen=True)
class CorrectionConfig:
    correction: str
    feature_fn: object
    alpha: float = 1e-4


def correction_configs():
    return {
        "ridge": CorrectionConfig("ridge", correction_features),
        "fsl_light": CorrectionConfig("fsl_light", fsl_light_features),
    }


def method_specs(base_method, fitted_corrections):
    def correction_for(name):
        if hasattr(fitted_corrections, "get"):
            return fitted_corrections.get(name)
        return fitted_corrections

    return [
        (base_method, "none", None, None),
        (f"{base_method}_plus_ridge", "ridge", correction_for("ridge"), correction_features),
        (f"{base_method}_plus_fsl_light", "fsl_light", correction_for("fsl_light"), fsl_light_features),
    ]


def fit_correction(data, train_curves, base_predictions, config):
    features, residual, names = stack_residual_training_rows(
        data, train_curves, base_predictions, config.feature_fn
    )
    return RidgeResidualCorrection(alpha=config.alpha).fit(features, residual, names)


def main():
    ensure_results_dirs()
    metric_rows = []
    stage_rows = []
    configs = correction_configs()

    for size in SIZES:
        data = load_size_data(size)
        tissue = fit_best_tissue(data, STRICT_COSINE_TRAIN_CURVES)
        mpl_params, _ = fit_mpl_scipy(data, STRICT_COSINE_TRAIN_CURVES, size)

        base_predictors = {
            "tissue": lambda curve: tissue["model"].predict_curve(data[curve]["lrs"], data[curve]["step"]),
            "mpl": lambda curve: predict_mpl_curve(data[curve]["lrs"], data[curve]["step"], mpl_params),
        }

        corrections = {}
        for base_method, predict in base_predictors.items():
            base_predictions = {curve: predict(curve) for curve in STRICT_COSINE_TRAIN_CURVES}
            corrections[base_method] = {
                name: fit_correction(data, STRICT_COSINE_TRAIN_CURVES, base_predictions, config)
                for name, config in configs.items()
            }

        for base_method, predict in base_predictors.items():
            for method, correction_name, correction, feature_fn in method_specs(base_method, corrections[base_method]):
                for curve in ALL_EVAL_CURVES:
                    base_pred = predict(curve)
                    if correction is None:
                        pred = base_pred
                    else:
                        features, _ = feature_fn(data[curve]["lrs"], data[curve]["step"])
                        pred = base_pred + correction.predict_residual(features)

                    metric_rows.append(
                        metric_row(
                            {
                                "run_id": RUN_ID,
                                "size": size,
                                "method": method,
                                "base_method": base_method,
                                "correction": correction_name,
                                "train_curves": ";".join(STRICT_COSINE_TRAIN_CURVES),
                                "test_curve": curve,
                            },
                            data[curve]["loss"],
                            pred,
                        )
                    )
                    stage_rows.extend(
                        stage_metric_rows(
                            {
                                "run_id": RUN_ID,
                                "size": size,
                                "method": method,
                                "test_curve": curve,
                            },
                            data[curve]["step"],
                            data[curve]["loss"],
                            pred,
                        )
                    )
                    write_prediction_csv(
                        RUN_ID,
                        size,
                        method,
                        curve,
                        prediction_rows(size, method, curve, data[curve]["step"], data[curve]["loss"], pred),
                    )

    write_csv(
        ROOT / "results" / "metrics" / "cross_schedule_metrics.csv",
        metric_rows,
        [
            "run_id",
            "size",
            "method",
            "base_method",
            "correction",
            "train_curves",
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
        ROOT / "results" / "metrics" / "fsl_light_metrics.csv",
        [row for row in metric_rows if row["correction"] == "fsl_light"],
        [
            "run_id",
            "size",
            "method",
            "base_method",
            "correction",
            "train_curves",
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
        ROOT / "results" / "metrics" / "stage_metrics.csv",
        stage_rows,
        [
            "run_id",
            "size",
            "method",
            "test_curve",
            "stage",
            "start_step",
            "end_step",
            "n_points",
            "rmse",
            "mae",
            "prede",
            "mean_signed_error",
        ],
    )


if __name__ == "__main__":
    main()
