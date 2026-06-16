# Direction C NCPL-Style Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Direction C from `docs/task2_improvement_plan_zh.html`: an NCPL-style neural surrogate that predicts loss directly from model size, step, and learning-rate schedule functionals.

**Architecture:** Keep Direction C separate from the existing Tissue/MPL residual-correction path. Add a focused surrogate module for feature construction and log-loss MLP fitting, a standalone experiment script for metrics/predictions, then integrate one comparison figure and README notes that frame this as a high-variance future-work baseline.

**Tech Stack:** Python, NumPy, scikit-learn `MLPRegressor`, existing CSV loaders, existing metrics and plotting helpers, pytest.

---

## File Structure

- Create `src/surrogate.py`: feature construction, train-row stacking, and `NCPLStyleSurrogate`.
- Create `tests/test_surrogate.py`: unit tests for feature names, shape, finite values, model-size signal, log-loss fitting/prediction guardrails.
- Create `scripts/run_ncpl_surrogate.py`: standalone Direction C experiment that writes metrics and predictions.
- Create `tests/test_ncpl_surrogate.py`: script-level tests for train/eval split behavior, output row keys, and prediction CSV naming.
- Modify `scripts/make_figures.py`: add NCPL method label and `plot_ncpl_surrogate_comparison()`.
- Modify `tests/test_figures.py`: test NCPL label, plot skip behavior, saved figure name, and `main()` call.
- Modify `README.md` and `README.zh.md`: document Direction C outputs and its intended interpretation.

## Task 1: Core NCPL-Style Surrogate Module

**Files:**
- Create: `src/surrogate.py`
- Create: `tests/test_surrogate.py`

- [ ] **Step 1: Write failing tests for feature construction**

```python
import numpy as np
import pytest

from src.surrogate import ncpl_surrogate_features, NCPLStyleSurrogate


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_surrogate.py::test_ncpl_surrogate_features_include_model_size_and_schedule_functionals -q
```

Expected: FAIL because `src.surrogate` does not exist.

- [ ] **Step 3: Implement `ncpl_surrogate_features()`**

Create `src/surrogate.py` with:

```python
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

from src.features import fsl_light_features


def _parse_model_size(model_size: str | int | float) -> float:
    text = str(model_size).strip().lower().replace("m", "")
    value = float(text)
    if value <= 0.0:
        raise ValueError("model_size must be positive.")
    return value


def ncpl_surrogate_features(
    model_size: str | int | float,
    lrs: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    steps = np.asarray(steps, dtype=int)
    lrs = np.asarray(lrs, dtype=float)
    fsl_features, fsl_names = fsl_light_features(lrs, steps)

    max_step = max(float(np.max(steps)) if len(steps) else 0.0, 1.0)
    step_norm = steps.astype(float) / max_step
    log_step = np.log1p(steps.astype(float))
    model_column = np.full(len(steps), np.log(_parse_model_size(model_size)), dtype=float)

    selected = ["log_tau", "eta_norm", "decay_conv", "is_decay"]
    fsl_columns = [fsl_features[:, fsl_names.index(name)] for name in selected]
    features = np.column_stack([model_column, step_norm, log_step, *fsl_columns])
    names = ["log_model_size", "step_norm", "log_step", *selected]
    return features, names
```

- [ ] **Step 4: Write failing tests for fitting and prediction**

Append to `tests/test_surrogate.py`:

```python
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


def test_ncpl_surrogate_rejects_prediction_before_fit():
    model = NCPLStyleSurrogate(hidden_layer_sizes=(4,), max_iter=5, random_state=0)

    with pytest.raises(ValueError, match="must be fitted"):
        model.predict(np.ones((2, 3)))
```

- [ ] **Step 5: Run the new tests to verify they fail for the missing class**

Run:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_surrogate.py -q
```

Expected: FAIL because `NCPLStyleSurrogate` is not implemented.

- [ ] **Step 6: Implement `NCPLStyleSurrogate` and row stacking**

Append to `src/surrogate.py`:

```python
class NCPLStyleSurrogate:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        alpha: float = 1e-3,
        max_iter: int = 2000,
        random_state: int = 0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.feature_names_: list[str] | None = None
        self.model_ = None

    def fit(self, features: np.ndarray, loss: np.ndarray, feature_names: list[str]):
        loss = np.asarray(loss, dtype=float)
        if np.any(loss <= 0.0):
            raise ValueError("NCPLStyleSurrogate requires positive loss values.")
        self.feature_names_ = list(feature_names)
        self.model_ = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
                learning_rate_init=1e-3,
            ),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_.fit(features, np.log(loss))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("NCPLStyleSurrogate must be fitted before prediction.")
        return np.exp(self.model_.predict(features))


def stack_surrogate_training_rows(data_by_size: dict, train_curves: list[str]):
    feature_blocks = []
    loss_blocks = []
    feature_names = None
    for size, data in data_by_size.items():
        for curve in train_curves:
            features, names = ncpl_surrogate_features(size, data[curve]["lrs"], data[curve]["step"])
            if feature_names is None:
                feature_names = names
            feature_blocks.append(features)
            loss_blocks.append(data[curve]["loss"])
    return np.vstack(feature_blocks), np.concatenate(loss_blocks), feature_names
```

- [ ] **Step 7: Run focused surrogate tests**

Run:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_surrogate.py -q
```

Expected: PASS.

## Task 2: Direction C Experiment Script and Output CSVs

**Files:**
- Create: `scripts/run_ncpl_surrogate.py`
- Create: `tests/test_ncpl_surrogate.py`

- [ ] **Step 1: Write failing script-level tests**

Create `tests/test_ncpl_surrogate.py`:

```python
import numpy as np

from scripts import run_ncpl_surrogate


def test_build_data_by_size_loads_each_requested_size(monkeypatch):
    calls = []

    def fake_load(size):
        calls.append(size)
        return {"cosine_24000.csv": {"loss": np.array([1.0])}}

    monkeypatch.setattr(run_ncpl_surrogate, "load_size_data", fake_load)

    result = run_ncpl_surrogate.build_data_by_size(["25", "100"])

    assert calls == ["25", "100"]
    assert sorted(result) == ["100", "25"]


def test_evaluate_surrogate_writes_metric_stage_and_prediction_rows(monkeypatch):
    data_by_size = {
        "25": {
            "wsd_20000_24000.csv": {
                "lrs": np.array([0.2, 0.1]),
                "step": np.array([0, 1]),
                "loss": np.array([2.0, 1.0]),
            }
        }
    }

    class ConstantSurrogate:
        def predict(self, features):
            return np.array([2.0, 1.0])

    predictions = []
    monkeypatch.setattr(run_ncpl_surrogate, "write_prediction_csv", lambda *args: predictions.append(args))

    metric_rows, stage_rows = run_ncpl_surrogate.evaluate_surrogate(
        surrogate=ConstantSurrogate(),
        data_by_size=data_by_size,
        eval_curves=["wsd_20000_24000.csv"],
    )

    assert metric_rows[0]["method"] == "ncpl_surrogate"
    assert metric_rows[0]["correction"] == "none"
    assert metric_rows[0]["train_curves"] == "cosine_24000.csv"
    assert stage_rows
    assert predictions[0][0] == run_ncpl_surrogate.RUN_ID
    assert predictions[0][2] == "ncpl_surrogate"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_ncpl_surrogate.py -q
```

Expected: FAIL because `scripts.run_ncpl_surrogate` does not exist.

- [ ] **Step 3: Implement script functions and `main()`**

Create `scripts/run_ncpl_surrogate.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_utils import (
    ensure_results_dirs,
    load_size_data,
    metric_row,
    prediction_rows,
    stage_metric_rows,
    write_csv,
    write_prediction_csv,
)
from src.splits import ALL_EVAL_CURVES, SIZES, STRICT_COSINE_TRAIN_CURVES
from src.surrogate import NCPLStyleSurrogate, ncpl_surrogate_features, stack_surrogate_training_rows


RUN_ID = "ncpl_surrogate_cosine_train"
METHOD = "ncpl_surrogate"


def build_data_by_size(sizes):
    return {size: load_size_data(size) for size in sizes}


def fit_surrogate(data_by_size, train_curves):
    features, loss, names = stack_surrogate_training_rows(data_by_size, train_curves)
    return NCPLStyleSurrogate().fit(features, loss, names)


def evaluate_surrogate(surrogate, data_by_size, eval_curves):
    metric_rows = []
    stage_rows = []
    train_curves = ";".join(STRICT_COSINE_TRAIN_CURVES)
    for size, data in data_by_size.items():
        for curve in eval_curves:
            features, _ = ncpl_surrogate_features(size, data[curve]["lrs"], data[curve]["step"])
            pred = surrogate.predict(features)
            base = {
                "run_id": RUN_ID,
                "size": size,
                "method": METHOD,
                "base_method": "none",
                "correction": "none",
                "train_curves": train_curves,
                "test_curve": curve,
            }
            metric_rows.append(metric_row(base, data[curve]["loss"], pred))
            stage_rows.extend(stage_metric_rows(base, data[curve]["step"], data[curve]["loss"], pred))
            write_prediction_csv(
                RUN_ID,
                size,
                METHOD,
                curve,
                prediction_rows(size, METHOD, curve, data[curve]["step"], data[curve]["loss"], pred),
            )
    return metric_rows, stage_rows


def main():
    ensure_results_dirs()
    data_by_size = build_data_by_size(SIZES)
    surrogate = fit_surrogate(data_by_size, STRICT_COSINE_TRAIN_CURVES)
    metric_rows, stage_rows = evaluate_surrogate(surrogate, data_by_size, ALL_EVAL_CURVES)
    metric_fields = [
        "run_id",
        "size",
        "method",
        "base_method",
        "correction",
        "train_curves",
        "test_curve",
        "rmse",
        "mae",
        "prede",
        "worste",
        "r2",
        "final_rel_error",
        "auc_rel_error",
    ]
    write_csv(ROOT / "results" / "metrics" / "ncpl_surrogate_metrics.csv", metric_rows, metric_fields)
    write_csv(
        ROOT / "results" / "metrics" / "ncpl_surrogate_stage_metrics.csv",
        stage_rows,
        [
            "run_id",
            "size",
            "method",
            "base_method",
            "correction",
            "train_curves",
            "test_curve",
            "stage",
            "start_step",
            "end_step",
            "n_points",
            "rmse",
            "mae",
            "prede",
            "mean_signed_error",
        ],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused script tests**

Run:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_ncpl_surrogate.py tests/test_surrogate.py -q
```

Expected: PASS.

## Task 3: Figures and Documentation Integration

**Files:**
- Modify: `scripts/make_figures.py`
- Modify: `tests/test_figures.py`
- Modify: `README.md`
- Modify: `README.zh.md`

- [ ] **Step 1: Write failing figure tests**

Append to `tests/test_figures.py`:

```python
def test_method_label_includes_ncpl_surrogate():
    assert make_figures.method_label("ncpl_surrogate") == "NCPL-style surrogate"


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
```

Update `test_main_calls_direction_b_plot_functions()` by adding `"plot_ncpl_surrogate_comparison"` to the expected `plot_functions` list.

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_figures.py -q
```

Expected: FAIL because `method_label()` and `main()` do not include NCPL.

- [ ] **Step 3: Implement plotting support**

Modify `scripts/make_figures.py`:

```python
NCPL_METHODS = ["ncpl_surrogate"]
```

Add to `method_label()`:

```python
"ncpl_surrogate": "NCPL-style surrogate",
```

Add:

```python
def plot_ncpl_surrogate_comparison():
    rows = [
        row
        for row in read_rows(METRICS_DIR / "ncpl_surrogate_metrics.csv")
        if row["test_curve"] in MAIN_TEST_CURVES and row["method"] in NCPL_METHODS
    ]
    if not rows:
        return

    labels = [
        f"{row['size']}M\n{method_label(row['method'])}\n{row['test_curve'].split('_')[0].upper()}"
        for row in rows
    ]
    values = [float(row["rmse"]) for row in rows]
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("RMSE")
    plt.title("Direction C: NCPL-style direct surrogate on WSD/WSDLD")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ncpl_surrogate_comparison.png", dpi=160)
    plt.close()
```

Call `plot_ncpl_surrogate_comparison()` from `main()`.

- [ ] **Step 4: Update documentation**

In `README.md`, add a concise Direction C section:

```markdown
### Direction C: NCPL-style surrogate

Direction C adds `scripts/run_ncpl_surrogate.py`, a direct neural surrogate that uses model size, step progress, and learning-rate schedule functionals to predict log loss. It writes:

- `results/metrics/ncpl_surrogate_metrics.csv`
- `results/metrics/ncpl_surrogate_stage_metrics.csv`
- `results/predictions/ncpl_surrogate_cosine_train_<size>_ncpl_surrogate_<curve>.csv`
- `results/figures/ncpl_surrogate_comparison.png`

This is intentionally reported as a high-risk baseline/future-work direction because the course dataset has only a small number of schedules.
```

Add the equivalent Chinese note to `README.zh.md`.

- [ ] **Step 5: Run focused figure tests**

Run:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_figures.py -q
```

Expected: PASS.

## Final Verification

- [ ] Run focused tests:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_surrogate.py tests/test_ncpl_surrogate.py tests/test_figures.py -q
```

- [ ] Run full tests:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest
```

- [ ] Run Direction C experiment:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_ncpl_surrogate.py
```

- [ ] Regenerate figures:

```powershell
& 'C:\Users\27854\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/make_figures.py
```

- [ ] Confirm outputs exist:

```powershell
Test-Path results\metrics\ncpl_surrogate_metrics.csv
Test-Path results\metrics\ncpl_surrogate_stage_metrics.csv
Test-Path results\figures\ncpl_surrogate_comparison.png
```
