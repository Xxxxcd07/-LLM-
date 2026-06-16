import numpy as np
import pytest

from src.annealing_law import TissueAnnealingLaw, fit_best_tissue
from src.correction import RidgeResidualCorrection, stack_residual_training_rows


def test_fit_best_tissue_on_synthetic_power_curve():
    lrs = np.full(20, 0.1)
    steps = np.arange(1, 20)
    s1 = np.cumsum(lrs)[steps]
    loss = 1.0 + 0.5 * s1 ** -0.5
    data = {"constant.csv": {"lrs": lrs, "step": steps, "loss": loss}}
    best = fit_best_tissue(data, ["constant.csv"], lambdas=(0.99,))
    pred = best["model"].predict_curve(lrs, steps)
    assert np.mean(np.abs(pred - loss)) < 0.02


def test_fit_best_tissue_on_synthetic_decay_curve():
    lrs = np.array([0.2] * 8 + [0.1] * 8 + [0.05] * 8, dtype=float)
    steps = np.arange(1, len(lrs))
    source = TissueAnnealingLaw(lambda_decay=0.5)
    loss = source.predict_curve(lrs, steps, params=[1.0, 0.4, 0.5, 0.2])
    data = {"decay.csv": {"lrs": lrs, "step": steps, "loss": loss}}
    best = fit_best_tissue(data, ["decay.csv"], lambdas=(0.5,))
    pred = best["model"].predict_curve(lrs, steps)
    assert best["lambda_decay"] == 0.5
    assert np.mean(np.abs(pred - loss)) < 0.03


def test_ridge_residual_correction_recovers_linear_residual():
    x = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
    y = 2.0 + 0.5 * np.arange(5, dtype=float)
    model = RidgeResidualCorrection(alpha=0.0).fit(x, y, ["bias", "x"])
    pred = model.predict_residual(x)
    assert np.allclose(pred, y)


def test_ridge_residual_correction_requires_fit_before_prediction():
    model = RidgeResidualCorrection()
    features = np.ones((2, 2), dtype=float)
    with pytest.raises(ValueError, match="must be fitted"):
        model.predict_residual(features)


def test_stack_residual_training_rows_preserves_order_and_names():
    data = {
        "first.csv": {
            "lrs": np.array([0.1, 0.2, 0.3]),
            "step": np.array([0, 2]),
            "loss": np.array([3.0, 2.0]),
        },
        "second.csv": {
            "lrs": np.array([0.4, 0.5]),
            "step": np.array([1]),
            "loss": np.array([1.5]),
        },
    }
    base_predictions = {
        "first.csv": np.array([2.5, 1.0]),
        "second.csv": np.array([1.0]),
    }

    def feature_fn(lrs, steps):
        features = np.column_stack([np.ones(len(steps)), lrs[steps]])
        return features, ["bias", "lr_at_step"]

    features, residual, names = stack_residual_training_rows(
        data, ["first.csv", "second.csv"], base_predictions, feature_fn
    )

    expected_features = np.array(
        [
            [1.0, 0.1],
            [1.0, 0.3],
            [1.0, 0.5],
        ]
    )
    expected_residual = np.array([0.5, 1.0, 0.5])
    assert np.allclose(features, expected_features)
    assert np.allclose(residual, expected_residual)
    assert names == ["bias", "lr_at_step"]
