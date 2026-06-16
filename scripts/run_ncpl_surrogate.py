import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_utils import (
    ensure_results_dirs,
    load_size_data,
    metric_row,
    prediction_rows,
    stage_metric_rows,
    write_csv,
    write_prediction_csv,
)
from src.splits import ALL_EVAL_CURVES, SIZES, STRICT_COSINE_TRAIN_CURVES
from src.surrogate import (
    NCPLStyleSurrogate,
    ncpl_surrogate_features,
    stack_surrogate_training_rows,
)


RUN_ID = "ncpl_surrogate_cosine_train"
METHOD = "ncpl_surrogate"


METRIC_FIELDS = [
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
]

STAGE_FIELDS = [
    "run_id",
    "size",
    "method",
    "base_method",
    "correction",
    "train_curves",
    "test_curve",
    "stage",
    "start_step",
    "end_step",
    "n_points",
    "rmse",
    "mae",
    "prede",
    "mean_signed_error",
]


def build_data_by_size(sizes):
    return {size: load_size_data(size) for size in sizes}


def fit_surrogate(data_by_size, train_curves):
    features, loss, names = stack_surrogate_training_rows(data_by_size, train_curves)
    return NCPLStyleSurrogate().fit(features, loss, names)


def evaluate_surrogate(surrogate, data_by_size, eval_curves):
    metric_rows = []
    stage_rows = []
    train_curves = ";".join(STRICT_COSINE_TRAIN_CURVES)
    for size, data in data_by_size.items():
        for curve in eval_curves:
            features, _ = ncpl_surrogate_features(
                size,
                data[curve]["lrs"],
                data[curve]["step"],
            )
            pred = surrogate.predict(features)
            base = {
                "run_id": RUN_ID,
                "size": size,
                "method": METHOD,
                "base_method": "none",
                "correction": "none",
                "train_curves": train_curves,
                "test_curve": curve,
            }
            metric_rows.append(metric_row(base, data[curve]["loss"], pred))
            stage_rows.extend(
                stage_metric_rows(
                    base,
                    data[curve]["step"],
                    data[curve]["loss"],
                    pred,
                )
            )
            write_prediction_csv(
                RUN_ID,
                size,
                METHOD,
                curve,
                prediction_rows(
                    size,
                    METHOD,
                    curve,
                    data[curve]["step"],
                    data[curve]["loss"],
                    pred,
                ),
            )
    return metric_rows, stage_rows


def main():
    ensure_results_dirs()
    data_by_size = build_data_by_size(SIZES)
    surrogate = fit_surrogate(data_by_size, STRICT_COSINE_TRAIN_CURVES)
    metric_rows, stage_rows = evaluate_surrogate(
        surrogate,
        data_by_size,
        ALL_EVAL_CURVES,
    )
    write_csv(
        ROOT / "results" / "metrics" / "ncpl_surrogate_metrics.csv",
        metric_rows,
        METRIC_FIELDS,
    )
    write_csv(
        ROOT / "results" / "metrics" / "ncpl_surrogate_stage_metrics.csv",
        stage_rows,
        STAGE_FIELDS,
    )


if __name__ == "__main__":
    main()
