import math

import pytest

from scripts import run_robustness


def test_stage_sensitivity_rows_pair_main_curve_stages_and_compute_decay_delta():
    stage_rows = [
        {
            "run_id": "cross_schedule_cosine_train",
            "size": "25",
            "method": "mpl",
            "test_curve": "wsd_20000_24000.csv",
            "stage": "post_warmup_stable",
            "rmse": "0.2",
        },
        {
            "run_id": "cross_schedule_cosine_train",
            "size": "25",
            "method": "mpl",
            "test_curve": "wsd_20000_24000.csv",
            "stage": "decay",
            "rmse": "0.5",
        },
        {
            "run_id": "cross_schedule_cosine_train",
            "size": "25",
            "method": "mpl",
            "test_curve": "wsdld_20000_24000.csv",
            "stage": "post_warmup_stable",
            "rmse": "0.0",
        },
        {
            "run_id": "cross_schedule_cosine_train",
            "size": "25",
            "method": "mpl",
            "test_curve": "wsdld_20000_24000.csv",
            "stage": "decay",
            "rmse": "0.1",
        },
        {
            "run_id": "cross_schedule_cosine_train",
            "size": "25",
            "method": "mpl",
            "test_curve": "constant_72000.csv",
            "stage": "decay",
            "rmse": "9.0",
        },
        {
            "run_id": "cross_schedule_cosine_train",
            "size": "25",
            "method": "mpl",
            "test_curve": "wsd_20000_24000.csv",
            "stage": "warmup_or_start",
            "rmse": "7.0",
        },
    ]

    rows = run_robustness.stage_sensitivity_rows(stage_rows)

    assert [(row["test_curve"], row["stable_rmse"], row["decay_rmse"]) for row in rows] == [
        ("wsd_20000_24000.csv", 0.2, 0.5),
        ("wsdld_20000_24000.csv", 0.0, 0.1),
    ]
    assert rows[0]["decay_minus_stable_rmse"] == pytest.approx(0.3)
    assert rows[0]["decay_to_stable_rmse_ratio"] == pytest.approx(2.5)
    assert math.isnan(rows[1]["decay_to_stable_rmse_ratio"])


def test_multistart_summary_rows_use_expected_columns_and_single_row_population_stats():
    metric_rows = [
        {
            "run_id": "direction_b_robustness",
            "size": "25",
            "start_id": "0",
            "test_curve": "wsd_20000_24000.csv",
            "objective": "1.5",
            "rmse": "0.2",
            "final_rel_error": "0.3",
            "auc_rel_error": "0.4",
        }
    ]

    summary = run_robustness.multistart_summary_rows(metric_rows)

    assert summary == [
        {
            "size": "25",
            "test_curve": "wsd_20000_24000.csv",
            "rmse_mean": 0.2,
            "rmse_std": 0.0,
            "rmse_min": 0.2,
            "rmse_max": 0.2,
            "final_rel_error_mean": 0.3,
            "final_rel_error_std": 0.0,
            "final_rel_error_min": 0.3,
            "final_rel_error_max": 0.3,
            "auc_rel_error_mean": 0.4,
            "auc_rel_error_std": 0.0,
            "auc_rel_error_min": 0.4,
            "auc_rel_error_max": 0.4,
            "objective_mean": 1.5,
            "objective_std": 0.0,
            "objective_min": 1.5,
            "objective_max": 1.5,
        }
    ]


def test_multistart_summary_rows_group_by_size_and_test_curve_across_starts():
    metric_rows = [
        {
            "size": "25",
            "start_id": "0",
            "test_curve": "wsd_20000_24000.csv",
            "objective": "1.0",
            "rmse": "0.2",
            "final_rel_error": "0.4",
            "auc_rel_error": "0.6",
        },
        {
            "size": "25",
            "start_id": "1",
            "test_curve": "wsd_20000_24000.csv",
            "objective": "3.0",
            "rmse": "0.6",
            "final_rel_error": "0.8",
            "auc_rel_error": "1.0",
        },
        {
            "size": "25",
            "start_id": "0",
            "test_curve": "wsdld_20000_24000.csv",
            "objective": "2.0",
            "rmse": "0.3",
            "final_rel_error": "0.5",
            "auc_rel_error": "0.7",
        },
        {
            "size": "100",
            "start_id": "0",
            "test_curve": "wsd_20000_24000.csv",
            "objective": "4.0",
            "rmse": "0.9",
            "final_rel_error": "1.1",
            "auc_rel_error": "1.3",
        },
    ]

    summary = run_robustness.multistart_summary_rows(metric_rows)

    assert [(row["size"], row["test_curve"]) for row in summary] == [
        ("100", "wsd_20000_24000.csv"),
        ("25", "wsd_20000_24000.csv"),
        ("25", "wsdld_20000_24000.csv"),
    ]

    two_start_group = summary[1]
    assert two_start_group["rmse_mean"] == pytest.approx(0.4)
    assert two_start_group["rmse_std"] == pytest.approx(0.2)
    assert two_start_group["rmse_min"] == pytest.approx(0.2)
    assert two_start_group["rmse_max"] == pytest.approx(0.6)
    assert two_start_group["final_rel_error_mean"] == pytest.approx(0.6)
    assert two_start_group["final_rel_error_std"] == pytest.approx(0.2)
    assert two_start_group["auc_rel_error_mean"] == pytest.approx(0.8)
    assert two_start_group["auc_rel_error_std"] == pytest.approx(0.2)
    assert two_start_group["objective_mean"] == pytest.approx(2.0)
    assert two_start_group["objective_std"] == pytest.approx(1.0)

    one_start_other_curve = summary[2]
    assert one_start_other_curve["rmse_mean"] == pytest.approx(0.3)
    assert one_start_other_curve["rmse_std"] == pytest.approx(0.0)
    assert one_start_other_curve["objective_min"] == pytest.approx(2.0)
    assert one_start_other_curve["objective_max"] == pytest.approx(2.0)


def test_main_requires_stage_metrics_before_writing_stage_sensitivity(monkeypatch, tmp_path):
    writes = []

    def fail_if_called(size, data):
        raise AssertionError("multistart should not run without stage_metrics.csv")

    monkeypatch.setattr(run_robustness, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(run_robustness, "SIZES", ["25"])
    monkeypatch.setattr(run_robustness, "load_size_data", lambda size: {"size": size})
    monkeypatch.setattr(run_robustness, "multistart_metric_rows", fail_if_called)
    monkeypatch.setattr(run_robustness, "write_csv", lambda path, rows, fieldnames: writes.append(path.name))

    with pytest.raises(FileNotFoundError, match=r"python scripts/run_cross_schedule\.py"):
        run_robustness.main()

    assert "stage_sensitivity_metrics.csv" not in writes
