import numpy as np

from src.features import (
    positive_decay,
    tissue_s2,
    correction_features,
    fsl_light_features,
)
from src.splits import phase_mask


def test_phase_mask_skips_empty_warmup_when_steps_start_after_warmup():
    steps = np.array([2160, 2176, 20000, 23936, 71936])
    mask = phase_mask(steps, "warmup_or_start")
    assert mask.sum() == 0


def test_phase_mask_decay_contains_decay_steps_only():
    steps = np.array([2160, 19984, 20000, 23936, 24000, 71936])
    mask = phase_mask(steps, "decay")
    assert steps[mask].tolist() == [20000, 23936]


def test_phase_mask_long_horizon_includes_boundary_step():
    steps = np.array([23936, 24000, 71936])
    mask = phase_mask(steps, "long_horizon")
    assert steps[mask].tolist() == [24000, 71936]


def test_positive_decay_ignores_lr_increases():
    lrs = np.array([0.0, 0.3, 0.2, 0.25, 0.1])
    result = positive_decay(lrs)
    assert np.allclose(result, [0.0, 0.0, 0.1, 0.0, 0.15])


def test_tissue_s2_accumulates_decay_memory():
    lrs = np.array([0.3, 0.2, 0.2, 0.1])
    result = tissue_s2(lrs, 0.5)
    assert np.allclose(result, [0.0, 0.1, 0.15, 0.275])


def test_correction_features_returns_rows_at_observed_steps():
    lrs = np.array([0.0, 0.1, 0.2, 0.1, 0.05])
    steps = np.array([2, 4])
    features, names = correction_features(lrs, steps)
    assert features.shape == (2, len(names))
    assert names[0] == "bias"


def test_fsl_light_features_returns_expected_names_and_shape():
    lrs = np.array([0.0, 0.1, 0.2, 0.1, 0.05])
    steps = np.array([2, 4])
    features, names = fsl_light_features(lrs, steps)

    assert names == ["bias", "log_tau", "eta_norm", "decay_conv", "is_decay"]
    assert features.shape == (len(steps), 5)


def test_fsl_light_log_tau_is_normalized_monotonic_and_finite():
    lrs = np.array([0.1, 0.2, 0.2, 0.1])
    steps = np.arange(len(lrs))
    features, names = fsl_light_features(lrs, steps)
    log_tau = features[:, names.index("log_tau")]

    assert np.all(np.isfinite(log_tau))
    assert np.all(np.diff(log_tau) >= 0.0)
    expected_tau = np.cumsum(lrs) / max(float(np.cumsum(lrs)[-1]), 1e-12)
    assert np.allclose(log_tau, np.log(np.maximum(expected_tau, 1e-12)))


def test_fsl_light_all_zero_lr_keeps_protected_columns_finite_and_zero():
    lrs = np.zeros(4)
    steps = np.arange(len(lrs))
    features, names = fsl_light_features(lrs, steps)

    log_tau = features[:, names.index("log_tau")]
    eta_norm = features[:, names.index("eta_norm")]
    decay_conv = features[:, names.index("decay_conv")]

    assert np.all(np.isfinite(log_tau))
    assert np.allclose(eta_norm, 0.0)
    assert np.allclose(decay_conv, 0.0)


def test_fsl_light_tiny_positive_lr_normalizes_terminal_tau_to_one():
    lrs = np.array([1e-20, 2e-20, 3e-20])
    steps = np.array([len(lrs) - 1])
    features, names = fsl_light_features(lrs, steps)

    assert features[0, names.index("log_tau")] == 0.0


def test_fsl_light_decay_conv_responds_to_lr_drop():
    lrs = np.array([0.3, 0.3, 0.1, 0.1])
    steps = np.arange(len(lrs))
    features, names = fsl_light_features(lrs, steps, lambda_decay=0.5)
    decay_conv = features[:, names.index("decay_conv")]
    is_decay = features[:, names.index("is_decay")]

    assert decay_conv[1] == 0.0
    assert decay_conv[2] > decay_conv[1]
    assert decay_conv[3] > decay_conv[2]
    assert np.allclose(is_decay, [0.0, 0.0, 1.0, 1.0])


def test_fsl_light_has_fewer_columns_than_full_correction_features():
    lrs = np.array([0.0, 0.1, 0.2, 0.1, 0.05])
    steps = np.array([2, 4])
    light, light_names = fsl_light_features(lrs, steps)
    full, full_names = correction_features(lrs, steps)

    assert full_names == [
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
    ]
    assert light.shape[1] == len(light_names)
    assert full.shape[1] == len(full_names)
    assert light.shape[1] < full.shape[1]
