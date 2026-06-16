import numpy as np

from scripts import run_ablation
from src.features import correction_features, fsl_light_features


def test_fsl_light_feature_set_uses_fsl_light_feature_names():
    lrs = np.array([0.2, 0.2, 0.1, 0.1])
    steps = np.arange(len(lrs))

    feature_fn = run_ablation.make_feature_fn(run_ablation.FEATURE_SETS["fsl_light"])
    features, names = feature_fn(lrs, steps)
    expected_features, expected_names = fsl_light_features(lrs, steps)

    assert names == expected_names
    assert np.allclose(features, expected_features)


def test_full_feature_set_preserves_full_ridge_feature_names():
    lrs = np.array([0.2, 0.2, 0.1, 0.1])
    steps = np.arange(len(lrs))

    feature_fn = run_ablation.make_feature_fn(run_ablation.FEATURE_SETS["full"])
    features, names = feature_fn(lrs, steps)
    expected_features, expected_names = correction_features(lrs, steps)

    assert names == expected_names
    assert np.allclose(features, expected_features)


def test_alpha_sweep_alphas_are_expected_values():
    assert run_ablation.ALPHA_SWEEP == [1e-6, 1e-4, 1e-2, 1.0]


def test_evaluate_correction_includes_alpha_and_unique_keys(monkeypatch):
    monkeypatch.setattr(run_ablation, "MAIN_TEST_CURVES", ["curve_a.csv", "curve_b.csv"])
    data = {
        "curve_a.csv": {
            "lrs": np.array([0.2, 0.1, 0.1]),
            "step": np.array([0, 1, 2]),
            "loss": np.array([3.0, 2.0, 1.0]),
        },
        "curve_b.csv": {
            "lrs": np.array([0.3, 0.3, 0.1]),
            "step": np.array([0, 1, 2]),
            "loss": np.array([4.0, 3.0, 2.0]),
        },
    }

    class ZeroCorrection:
        def predict_residual(self, features):
            return np.zeros(features.shape[0])

    rows = run_ablation.evaluate_correction(
        data=data,
        predict=lambda curve: data[curve]["loss"].copy(),
        correction=ZeroCorrection(),
        feature_fn=fsl_light_features,
        size="125m",
        base_method="mpl",
        feature_set="fsl_light",
        alpha=1e-4,
    )

    keys = [
        (row["size"], row["base_method"], row["feature_set"], row["alpha"], row["test_curve"])
        for row in rows
    ]

    assert len(rows) == 2
    assert all(row["alpha"] == 1e-4 for row in rows)
    assert len(keys) == len(set(keys))
