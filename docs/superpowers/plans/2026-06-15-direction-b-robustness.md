# Direction B Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Direction B from `docs/task2_improvement_plan_zh.html`: robustness and uncertainty experiments around MPL multi-start fitting, ridge alpha sensitivity, and stage-wise sensitivity.

**Architecture:** Keep the existing Direction A / FSL-light pipeline intact. Add one robustness script for deterministic MPL multi-start and stage-sensitivity CSV outputs, extend figure generation for these outputs, and document the new reproduction commands. Existing `scripts/run_ablation.py` already writes `results/metrics/alpha_sweep_metrics.csv`, so Direction B should reuse it and add visualization/reporting around it.

**Tech Stack:** Python, NumPy, SciPy/scikit-learn metrics, matplotlib, pytest.

---

### Task 1: MPL Multi-Start Helpers

**Files:**
- Modify: `src/experiment_utils.py`
- Modify: `tests/test_experiment_utils.py`

- [ ] Add tests for deterministic MPL initializations.
- [ ] Add a helper that returns small deterministic variants around `PARAMS[size]`, including the exact base parameter vector.
- [ ] Add a helper that summarizes metric rows by group keys with mean, std, min, and max for selected numeric columns.
- [ ] Run `python -m pytest tests/test_experiment_utils.py -q`.

### Task 2: Robustness Metrics Script

**Files:**
- Create: `scripts/run_robustness.py`
- Modify: `tests/test_robustness.py`

- [ ] Write failing tests for stage sensitivity summarization from synthetic `stage_metrics.csv` rows.
- [ ] Write failing tests for multi-start summary column names and one-row grouping behavior.
- [ ] Implement `scripts/run_robustness.py`.
- [ ] The script must write:
  - `results/metrics/mpl_multistart_metrics.csv`
  - `results/metrics/mpl_multistart_summary.csv`
  - `results/metrics/stage_sensitivity_metrics.csv`
- [ ] Multi-start should fit MPL on `STRICT_COSINE_TRAIN_CURVES`, evaluate `MAIN_TEST_CURVES`, and report per-start curve metrics plus objective value.
- [ ] Stage sensitivity should compare `post_warmup_stable` and `decay` RMSE for WSD/WSDLD rows, reporting `decay_minus_stable_rmse` and `decay_to_stable_rmse_ratio`.
- [ ] Run `python -m pytest tests/test_robustness.py -q`.

### Task 3: Direction B Figures And Docs

**Files:**
- Modify: `scripts/make_figures.py`
- Modify: `tests/test_figures.py`
- Modify: `README.md`
- Modify: `README.zh.md`

- [ ] Add tests that `make_figures.main()` calls Direction B plotting functions.
- [ ] Add an alpha sweep plot from `results/metrics/alpha_sweep_metrics.csv`.
- [ ] Add an MPL multi-start uncertainty plot from `results/metrics/mpl_multistart_summary.csv`.
- [ ] Add a stage sensitivity plot from `results/metrics/stage_sensitivity_metrics.csv`.
- [ ] Save figures as:
  - `results/figures/alpha_sweep_sensitivity.png`
  - `results/figures/mpl_multistart_uncertainty.png`
  - `results/figures/stage_sensitivity.png`
- [ ] Update README files with `python scripts/run_robustness.py` and the new outputs.
- [ ] Run `python -m pytest tests/test_figures.py -q`.

### Task 4: Verification

- [ ] Run `python -m pytest tests -q`.
- [ ] Run `python scripts/run_ablation.py`.
- [ ] Run `python scripts/run_cross_schedule.py`.
- [ ] Run `python scripts/run_robustness.py`.
- [ ] Run `python scripts/make_figures.py`.
- [ ] Verify expected Direction B CSV and PNG outputs exist.
