# FSL-light Direction A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Direction A from `docs/task2_improvement_plan_zh.html`: a low-complexity FSL-light residual correction for cross-schedule loss-curve prediction.

**Architecture:** Keep Tissue/MPL as base predictors and train only residual corrections on `cosine_24000.csv`. Add a compact FSL-light feature set, compare it with the existing full ridge correction, and generate CSV/PNG outputs for method development reporting.

**Tech Stack:** Python, NumPy, SciPy/scikit-learn metrics, matplotlib, pytest.

---

### Task 1: FSL-light Features

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

- [ ] Add `fsl_light_features(lrs, steps, lambda_decay=0.995)` with names `bias`, `log_tau`, `eta_norm`, `decay_conv`, `is_decay`.
- [ ] Keep `correction_features()` unchanged for the existing full ridge baseline.
- [ ] Test shape, names, finite monotonic intrinsic time, decay response, and fewer columns than full ridge.
- [ ] Run `python -m pytest tests/test_features.py -q`.

### Task 2: Experiment Integration

**Files:**
- Modify: `scripts/run_ablation.py`
- Modify: `scripts/run_cross_schedule.py`
- Modify as needed: `src/experiment_utils.py`
- Add tests if a reusable helper is introduced.

- [ ] Add `fsl_light` as an ablation feature set.
- [ ] Train `fsl_light` residual correction on `STRICT_COSINE_TRAIN_CURVES`.
- [ ] Add `tissue_plus_fsl_light` and `mpl_plus_fsl_light` to cross-schedule metrics, stage metrics, and prediction CSVs.
- [ ] Write `results/metrics/fsl_light_metrics.csv` with only the FSL-light rows.
- [ ] Write `results/metrics/alpha_sweep_metrics.csv` for alphas `1e-6`, `1e-4`, `1e-2`, and `1`.

### Task 3: Figures And Documentation

**Files:**
- Modify: `scripts/make_figures.py`
- Modify: `README.md`
- Modify: `README.zh.md`

- [ ] Include FSL-light methods in prediction and summary plots.
- [ ] Generate `results/figures/fsl_light_comparison.png`.
- [ ] Generate `results/figures/stage_fsl_light_error.png`.
- [ ] Document that full ridge failure is an overfitting diagnosis, not a bug.
- [ ] Document FSL-light outputs and reproduction commands.

### Task 4: Verification

- [ ] Run `python -m pytest tests -q`.
- [ ] Run `python scripts/run_cross_schedule.py`.
- [ ] Run `python scripts/run_ablation.py`.
- [ ] Run `python scripts/make_figures.py`.
- [ ] Verify expected CSV and PNG outputs exist.
