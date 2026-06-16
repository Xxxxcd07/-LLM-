import pytest

from scripts import make_figures
from scripts.make_figures import summarize_stage_errors


def capture_saved_figure_names(monkeypatch):
    saved = []
    monkeypatch.setattr(make_figures.plt, "savefig", lambda path, **kwargs: saved.append(path.name))
    return saved


def test_plot_methods_include_fsl_light_corrections():
    assert "tissue_plus_fsl_light" in make_figures.PLOT_METHODS
    assert "mpl_plus_fsl_light" in make_figures.PLOT_METHODS


def test_main_and_fsl_comparison_methods_have_separate_scopes():
    assert make_figures.MAIN_COMPARISON_METHODS == ["tissue", "mpl", "mpl_plus_ridge"]
    assert make_figures.FSL_COMPARISON_METHODS == [
        "tissue",
        "tissue_plus_ridge",
        "tissue_plus_fsl_light",
        "mpl",
        "mpl_plus_ridge",
        "mpl_plus_fsl_light",
    ]
    assert set(make_figures.MAIN_COMPARISON_METHODS) < set(make_figures.FSL_COMPARISON_METHODS)


def test_method_label_uses_exact_display_names():
    assert make_figures.method_label("tissue") == "Tissue"
    assert make_figures.method_label("tissue_plus_ridge") == "Tissue + Ridge"
    assert make_figures.method_label("tissue_plus_fsl_light") == "Tissue + FSL-light"
    assert make_figures.method_label("mpl") == "MPL"
    assert make_figures.method_label("mpl_plus_ridge") == "MPL + Ridge"
    assert make_figures.method_label("mpl_plus_fsl_light") == "MPL + FSL-light"


def test_method_label_includes_ncpl_surrogate():
    assert make_figures.method_label("ncpl_surrogate") == "NCPL-style surrogate"


def test_summarize_stage_errors_keeps_all_sizes_and_averages_curves():
    rows = []
    for size in ["25", "100", "400"]:
        for curve, rmse in [("wsd_20000_24000.csv", 0.1), ("wsdld_20000_24000.csv", 0.3)]:
            rows.append(
                {
                    "size": size,
                    "method": "mpl",
                    "stage": "decay",
                    "test_curve": curve,
                    "rmse": str(rmse),
                }
            )

    summary = summarize_stage_errors(rows)

    assert [row["size"] for row in summary] == ["25", "100", "400"]
    assert [row["rmse"] for row in summary] == [0.2, 0.2, 0.2]


def test_stage_rows_filter_excludes_aux_curves_before_summarizing():
    rows = [
        {
            "size": "25",
            "method": "mpl",
            "stage": "decay",
            "test_curve": "wsd_20000_24000.csv",
            "rmse": "0.10",
        },
        {
            "size": "25",
            "method": "mpl",
            "stage": "decay",
            "test_curve": "wsdcon_3.csv",
            "rmse": "0.90",
        },
    ]

    summary = summarize_stage_errors(
        make_figures.filter_stage_rows(rows, make_figures.MAIN_COMPARISON_METHODS)
    )

    assert summary == [{"size": "25", "method": "mpl", "stage": "decay", "rmse": 0.1}]


def test_summarize_stage_errors_includes_fsl_light_methods_by_size_method_stage():
    rows = [
        {
            "size": "25",
            "method": "tissue_plus_fsl_light",
            "stage": "post_warmup_stable",
            "test_curve": "wsd_20000_24000.csv",
            "rmse": "0.10",
        },
        {
            "size": "25",
            "method": "tissue_plus_fsl_light",
            "stage": "post_warmup_stable",
            "test_curve": "wsdld_20000_24000.csv",
            "rmse": "0.30",
        },
        {
            "size": "25",
            "method": "mpl_plus_fsl_light",
            "stage": "decay",
            "test_curve": "wsd_20000_24000.csv",
            "rmse": "0.20",
        },
        {
            "size": "25",
            "method": "mpl_plus_fsl_light",
            "stage": "decay",
            "test_curve": "wsdld_20000_24000.csv",
            "rmse": "0.40",
        },
    ]

    summary = summarize_stage_errors(rows)

    assert {
        (row["size"], row["method"], row["stage"]): row["rmse"]
        for row in summary
        if row["method"].endswith("_plus_fsl_light")
    } == {
        ("25", "tissue_plus_fsl_light", "post_warmup_stable"): pytest.approx(0.2),
        ("25", "mpl_plus_fsl_light", "decay"): pytest.approx(0.3),
    }


def test_fsl_light_figure_functions_skip_empty_rows(monkeypatch):
    monkeypatch.setattr(make_figures, "read_rows", lambda path: [])

    make_figures.plot_fsl_light_comparison()
    make_figures.plot_stage_fsl_light_error()


def test_main_calls_direction_b_plot_functions(monkeypatch):
    calls = []
    plot_functions = [
        "plot_lr_schedules",
        "plot_prediction_diagnostics",
        "plot_main_comparison",
        "plot_final_error",
        "plot_fsl_light_comparison",
        "plot_stage_errors",
        "plot_stage_fsl_light_error",
        "plot_ablation",
        "plot_coefficients",
        "plot_alpha_sweep_sensitivity",
        "plot_mpl_multistart_uncertainty",
        "plot_stage_sensitivity",
        "plot_ncpl_surrogate_comparison",
    ]
    monkeypatch.setattr(make_figures, "ensure_results_dirs", lambda: calls.append("ensure_results_dirs"))
    for name in plot_functions:
        monkeypatch.setattr(make_figures, name, lambda name=name: calls.append(name))

    make_figures.main()

    assert calls == ["ensure_results_dirs", *plot_functions]


def test_direction_b_figure_functions_skip_empty_rows(monkeypatch):
    saved = capture_saved_figure_names(monkeypatch)
    monkeypatch.setattr(make_figures, "read_rows", lambda path: [])

    make_figures.plot_alpha_sweep_sensitivity()
    make_figures.plot_mpl_multistart_uncertainty()
    make_figures.plot_stage_sensitivity()
    make_figures.plot_ncpl_surrogate_comparison()

    assert saved == []


def test_ncpl_surrogate_comparison_saves_expected_figure(monkeypatch):
    saved = capture_saved_figure_names(monkeypatch)
    rows = [
        {
            "size": "25",
            "method": "ncpl_surrogate",
            "test_curve": "wsd_20000_24000.csv",
            "rmse": "0.12",
        }
    ]
    monkeypatch.setattr(
        make_figures,
        "read_rows",
        lambda path: rows if path.name == "ncpl_surrogate_metrics.csv" else [],
    )

    make_figures.plot_ncpl_surrogate_comparison()

    assert saved == ["ncpl_surrogate_comparison.png"]


def test_alpha_sweep_sensitivity_saves_expected_figure(monkeypatch):
    saved = capture_saved_figure_names(monkeypatch)
    rows = [
        {
            "size": "25",
            "base_method": "mpl",
            "alpha": "1e-06",
            "test_curve": "wsd_20000_24000.csv",
            "rmse": "0.10",
        },
        {
            "size": "25",
            "base_method": "mpl",
            "alpha": "1e-03",
            "test_curve": "wsd_20000_24000.csv",
            "rmse": "0.08",
        },
    ]
    monkeypatch.setattr(
        make_figures,
        "read_rows",
        lambda path: rows if path.name == "alpha_sweep_metrics.csv" else [],
    )

    make_figures.plot_alpha_sweep_sensitivity()

    assert saved == ["alpha_sweep_sensitivity.png"]


def test_mpl_multistart_uncertainty_saves_expected_figure(monkeypatch):
    saved = capture_saved_figure_names(monkeypatch)
    rows = [
        {
            "size": "25",
            "test_curve": "wsd_20000_24000.csv",
            "rmse_mean": "0.10",
            "rmse_std": "0.02",
        }
    ]
    monkeypatch.setattr(
        make_figures,
        "read_rows",
        lambda path: rows if path.name == "mpl_multistart_summary.csv" else [],
    )

    make_figures.plot_mpl_multistart_uncertainty()

    assert saved == ["mpl_multistart_uncertainty.png"]


def test_stage_sensitivity_saves_expected_figure(monkeypatch):
    saved = capture_saved_figure_names(monkeypatch)
    rows = [
        {
            "size": "25",
            "method": "mpl",
            "test_curve": "wsd_20000_24000.csv",
            "decay_minus_stable_rmse": "0.05",
        }
    ]
    monkeypatch.setattr(
        make_figures,
        "read_rows",
        lambda path: rows if path.name == "stage_sensitivity_metrics.csv" else [],
    )

    make_figures.plot_stage_sensitivity()

    assert saved == ["stage_sensitivity.png"]
