import math

import numpy as np

from src import experiment_utils
from src.config import PARAMS


def test_mpl_multistart_initializations_are_deterministic_and_include_base():
    assert hasattr(experiment_utils, "mpl_multistart_initializations")

    first = experiment_utils.mpl_multistart_initializations("25")
    second = experiment_utils.mpl_multistart_initializations("25")
    base = np.array(PARAMS["25"], dtype=float)

    assert len(first) > 1
    assert len(first) == len(second)
    np.testing.assert_allclose(first[0], base, rtol=0.0, atol=0.0)

    for candidate, repeated in zip(first, second):
        assert isinstance(candidate, np.ndarray)
        assert candidate.shape == base.shape
        assert np.all(candidate > 0.0)
        np.testing.assert_allclose(candidate, repeated, rtol=0.0, atol=0.0)

    assert any(not np.array_equal(candidate, base) for candidate in first[1:])


def test_summarize_numeric_rows_groups_sorted_and_computes_population_stats():
    assert hasattr(experiment_utils, "summarize_numeric_rows")

    rows = [
        {"size": "100", "method": "mpl", "rmse": "3.0", "mae": "nan"},
        {"size": "25", "method": "mpl", "rmse": "1.0", "mae": "2.0"},
        {"size": "25", "method": "mpl", "rmse": "3.0", "mae": "bad"},
        {"size": "25", "method": "ridge", "rmse": "5.0", "mae": "4.0"},
        {"size": "100", "method": "mpl", "rmse": "inf", "mae": ""},
    ]

    summary = experiment_utils.summarize_numeric_rows(rows, ["size", "method"], ["rmse", "mae"])

    assert [(row["size"], row["method"]) for row in summary] == [
        ("100", "mpl"),
        ("25", "mpl"),
        ("25", "ridge"),
    ]
    assert summary[1]["rmse_mean"] == 2.0
    assert summary[1]["rmse_std"] == 1.0
    assert summary[1]["rmse_min"] == 1.0
    assert summary[1]["rmse_max"] == 3.0
    assert summary[1]["mae_mean"] == 2.0
    assert summary[1]["mae_std"] == 0.0
    assert summary[2]["rmse_std"] == 0.0
    assert math.isnan(summary[0]["mae_mean"])
    assert math.isnan(summary[0]["mae_std"])
    assert math.isnan(summary[0]["mae_min"])
    assert math.isnan(summary[0]["mae_max"])
