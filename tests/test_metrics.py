import numpy as np

from src.metrics import curve_metrics, stage_metrics
from src.predictors import predict_mpl_curve


def test_predict_mpl_curve_returns_one_prediction_per_observed_step():
    lrs = np.full(10, 0.1)
    steps = np.array([1, 5, 9])
    params = [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5]
    pred = predict_mpl_curve(lrs, steps, params)
    assert pred.shape == steps.shape
    assert np.all(np.isfinite(pred))


def test_predict_mpl_curve_matches_reference_decay_term():
    lrs = np.array([0.3, 0.2, 0.2, 0.1], dtype=float)
    steps = np.array([1, 2, 3])
    params = [1.0, 0.5, 0.5, 2.0, 1.5, 0.7, 0.2]
    pred = predict_mpl_curve(lrs, steps, params)
    l0, a, alpha, b, c, beta, gamma = params
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs))
    lr_gap[1:] = np.diff(lrs)
    expected = []
    for step in steps:
        s1 = lr_sum[step]
        ld = 0.0
        if step > 0:
            scaled = lrs[1 : step + 1] ** (-gamma) * (lr_sum[step] - lr_sum[:step])
            ld = np.sum(lr_gap[1 : step + 1] * (1 - (1 + c * scaled) ** (-beta)))
        expected.append(l0 + a * max(s1, 1e-12) ** (-alpha) + b * ld)
    assert np.allclose(pred, expected)


def test_curve_metrics_exact_prediction_is_zero_error():
    loss = np.array([3.0, 2.5, 2.0])
    pred = loss.copy()
    metrics = curve_metrics(loss, pred)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["final_rel_error"] == 0.0


def test_curve_metrics_reports_relative_and_auc_errors():
    loss = np.array([2.0, 4.0, 8.0])
    pred = np.array([3.0, 2.0, 10.0])
    metrics = curve_metrics(loss, pred)
    assert np.isclose(metrics["prede"], (0.5 + 0.5 + 0.25) / 3)
    assert np.isclose(metrics["worste"], 0.5)
    assert np.isclose(metrics["final_rel_error"], 0.25)
    assert np.isclose(metrics["auc_rel_error"], 0.5 / 9.0)


def test_stage_metrics_records_nan_for_empty_stage():
    steps = np.array([2160, 20000, 23936])
    loss = np.array([3.0, 2.8, 2.7])
    pred = np.array([3.1, 2.9, 2.6])
    rows = stage_metrics(steps, loss, pred)
    warmup = [row for row in rows if row["stage"] == "warmup_or_start"][0]
    assert warmup["n_points"] == 0
    assert np.isnan(warmup["rmse"])


def test_stage_metrics_assigns_boundary_steps_to_expected_stages():
    steps = np.array([2159, 2160, 19999, 20000, 23999, 24000])
    loss = np.ones_like(steps, dtype=float)
    pred = loss.copy()
    rows = {row["stage"]: row for row in stage_metrics(steps, loss, pred)}
    assert rows["warmup_or_start"]["n_points"] == 1
    assert rows["post_warmup_stable"]["n_points"] == 2
    assert rows["decay"]["n_points"] == 2
    assert rows["long_horizon"]["n_points"] == 1
