import numpy as np
import pytest

from src.surrogate import (
    NCPLStyleSurrogate,
    ncpl_surrogate_features,
    stack_surrogate_training_rows,
)


def test_ncpl_surrogate_features_include_model_size_and_schedule_functionals():
    lrs = np.array([0.3, 0.3, 0.1, 0.1])
    steps = np.arange(len(lrs))

    features, names = ncpl_surrogate_features(model_size="100", lrs=lrs, steps=steps)

    assert names == [
        "log_model_size",
        "step_norm",
        "log_step",
        "log_tau",
        "eta_norm",
        "decay_conv",
        "is_decay",
    ]
    assert features.shape == (4, len(names))
    assert np.all(np.isfinite(features))
    assert np.allclose(features[:, names.index("log_model_size")], np.log(100.0))
    assert features[0, names.index("step_norm")] == 0.0
    assert features[-1, names.index("step_norm")] == 1.0
    assert features[2, names.index("is_decay")] == 1.0


def test_ncpl_surrogate_fit_predicts_positive_loss_values():
    lrs = np.full(6, 0.1)
    steps = np.arange(6)
    features, names = ncpl_surrogate_features("25", lrs, steps)
    loss = np.exp(2.0 - 0.01 * steps)

    model = NCPLStyleSurrogate(hidden_layer_sizes=(4,), max_iter=500, random_state=0)
    model.fit(features, loss, names)
    pred = model.predict(features)

    assert pred.shape == loss.shape
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0.0)
    assert model.feature_names_ == names


def test_ncpl_surrogate_fit_rejects_non_positive_loss_values():
    lrs = np.full(3, 0.1)
    steps = np.arange(3)
    features, names = ncpl_surrogate_features("25", lrs, steps)

    model = NCPLStyleSurrogate(hidden_layer_sizes=(4,), max_iter=5, random_state=0)

    with pytest.raises(ValueError, match="positive loss"):
        model.fit(features, np.array([1.0, 0.0, -1.0]), names)


def test_ncpl_surrogate_rejects_prediction_before_fit():
    model = NCPLStyleSurrogate(hidden_layer_sizes=(4,), max_iter=5, random_state=0)

    with pytest.raises(ValueError, match="must be fitted"):
        model.predict(np.ones((2, 3)))


def test_stack_surrogate_training_rows_preserves_size_and_curve_order():
    data_by_size = {
        "25": {
            "cosine.csv": {
                "lrs": np.array([0.2, 0.1]),
                "step": np.array([0, 1]),
                "loss": np.array([2.0, 1.5]),
            }
        },
        "100": {
            "cosine.csv": {
                "lrs": np.array([0.3, 0.3, 0.1]),
                "step": np.array([0, 1, 2]),
                "loss": np.array([3.0, 2.5, 2.0]),
            }
        },
    }

    features, loss, names = stack_surrogate_training_rows(data_by_size, ["cosine.csv"])

    assert names == [
        "log_model_size",
        "step_norm",
        "log_step",
        "log_tau",
        "eta_norm",
        "decay_conv",
        "is_decay",
    ]
    assert features.shape == (5, len(names))
    assert np.allclose(loss, np.array([2.0, 1.5, 3.0, 2.5, 2.0]))
    assert np.allclose(features[:2, names.index("log_model_size")], np.log(25.0))
    assert np.allclose(features[2:, names.index("log_model_size")], np.log(100.0))
