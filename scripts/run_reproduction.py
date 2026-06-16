import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annealing_law import fit_best_tissue
from src.config import PARAMS
from src.experiment_utils import ensure_results_dirs, fit_mpl_scipy, load_size_data, metric_row, write_csv
from src.predictors import predict_mpl_curve
from src.splits import ALL_EVAL_CURVES, REPRODUCTION_TRAIN_CURVES, SIZES


RUN_ID = "reproduction_paper_split"


def main():
    parser = argparse.ArgumentParser(description="Run paper-like Tissue/MPL reproduction experiments.")
    parser.add_argument(
        "--fit-mpl",
        action="store_true",
        help="Fit MPL with SciPy instead of using the repository's precomputed parameters.",
    )
    parser.add_argument(
        "--mpl-maxiter",
        type=int,
        default=300,
        help="Maximum SciPy optimizer iterations when --fit-mpl is enabled.",
    )
    args = parser.parse_args()

    ensure_results_dirs()
    rows = []

    for size in SIZES:
        data = load_size_data(size)
        tissue = fit_best_tissue(data, REPRODUCTION_TRAIN_CURVES)
        if args.fit_mpl:
            mpl_params, _ = fit_mpl_scipy(
                data, REPRODUCTION_TRAIN_CURVES, size, maxiter=args.mpl_maxiter
            )
            protocol = "paper_like_train_split_fit_mpl"
        else:
            mpl_params = PARAMS[size]
            protocol = "paper_like_train_split_precomputed_mpl"
        methods = {
            "tissue": lambda curve: tissue["model"].predict_curve(data[curve]["lrs"], data[curve]["step"]),
            "mpl": lambda curve: predict_mpl_curve(data[curve]["lrs"], data[curve]["step"], mpl_params),
        }

        for method, predict in methods.items():
            for curve in ALL_EVAL_CURVES:
                pred = predict(curve)
                rows.append(
                    metric_row(
                        {
                            "run_id": RUN_ID,
                            "size": size,
                            "protocol": protocol,
                            "method": method,
                            "train_curves": ";".join(REPRODUCTION_TRAIN_CURVES),
                            "test_curve": curve,
                        },
                        data[curve]["loss"],
                        pred,
                    )
                )

    write_csv(
        ROOT / "results" / "metrics" / "reproduction_metrics.csv",
        rows,
        [
            "run_id",
            "size",
            "protocol",
            "method",
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


if __name__ == "__main__":
    main()
