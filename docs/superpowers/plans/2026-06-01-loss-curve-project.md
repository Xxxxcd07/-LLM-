# Loss Curve Project 实现计划

> **给 agentic workers 的要求：** 实现本计划时必须使用 `superpowers:subagent-driven-development`，推荐；或使用 `superpowers:executing-plans`，逐任务执行。所有步骤使用 checkbox 语法，方便逐项追踪。

**目标：** 完成课程 Task 2 的可复现实验项目：比较 Tissue et al. 的学习率退火律和 Luo et al. 的 Multi-Power Law 基线在 cosine-to-WSD 损失曲线预测上的表现，诊断分阶段误差，并评估一个可解释的 decay-aware residual correction。

**架构：** 保留现有 MultiPowerLaw 代码作为 MPL 基线，在此基础上增加若干职责单一的小模块：实验划分、schedule 特征、Tissue baseline、统一指标、残差修正和实验脚本。所有预测、指标和图表统一写入 `results/`，确保 GitHub 仓库和最终 slides 可以通过少量命令复现。

**技术栈：** Python 3.8+、NumPy、PyTorch、SciPy、scikit-learn、matplotlib，以及标准库中的 `csv/json`。

---

## 项目根目录

以下目录作为实验项目根目录：

```text
MultiPowerLaw-main/MultiPowerLaw-main
```

下文所有路径都相对于该目录。

## 范围

本项目拆成四个实现块：

1. 可复现的实验框架和固定 train/test split。
2. Tissue et al. 与 Luo et al. MPL 的 baseline reproduction。
3. Stage-wise error diagnosis 与 decay-aware residual correction。
4. 面向 slides 的指标表和图表产出。

本计划不训练新的 LLM，只使用 `loss_curve_repo/` 中已经提供的损失曲线。

## 计划文件结构

新增：

```text
src/splits.py
src/features.py
src/annealing_law.py
src/predictors.py
src/metrics.py
src/correction.py
scripts/run_reproduction.py
scripts/run_cross_schedule.py
scripts/run_ablation.py
scripts/make_figures.py
tests/test_features.py
tests/test_metrics.py
tests/test_correction.py
results/metrics/.gitkeep
results/predictions/.gitkeep
results/figures/.gitkeep
```

修改：

```text
README.md
requirements.txt
```

不要修改：

```text
loss_curve_repo/
25M/
100M/
400M/
optimized_schedules/
```

复用现有模块：

```text
src/data_loader.py
src/lr_schedules.py
src/models.py
src/fitting.py
src/evaluation.py
src/utils.py
src/config.py
```

## 实验协议

模型规模：

```text
25, 100, 400
```

主训练集：

```text
cosine_24000.csv
```

复现实验的 sanity check 训练集：

```text
constant_24000.csv
wsdcon_9.csv
```

主测试集：

```text
wsd_20000_24000.csv
wsdld_20000_24000.csv
```

额外泛化测试集：

```text
constant_72000.csv
cosine_72000.csv
wsdcon_3.csv
wsdcon_18.csv
```

阶段划分：

```text
warmup_or_start: step < 2160
post_warmup_stable: 2160 <= step < 20000
decay: 20000 <= step < 24000
long_horizon: step >= 24000
```

注意：提供的 CSV 大多从 step 2160 附近开始，因此 warmup 阶段可能没有观测点。指标代码应跳过空阶段，并将该阶段指标记录为 `nan`。

## 指标表产出

`results/metrics/reproduction_metrics.csv`

```text
run_id,size,protocol,method,train_curves,test_curve,rmse,mae,prede,worste,r2,final_rel_error,auc_rel_error
```

`results/metrics/cross_schedule_metrics.csv`

```text
run_id,size,method,base_method,correction,train_curves,test_curve,rmse,mae,prede,worste,r2,final_rel_error,auc_rel_error
```

`results/metrics/stage_metrics.csv`

```text
run_id,size,method,test_curve,stage,start_step,end_step,n_points,rmse,mae,prede,mean_signed_error
```

`results/metrics/ablation_metrics.csv`

```text
run_id,size,base_method,feature_set,test_curve,rmse,mae,prede,final_rel_error,auc_rel_error
```

`results/metrics/correction_coefficients.csv`

```text
run_id,size,base_method,feature_name,coefficient
```

## 图表产出

slides 主要图：

```text
results/figures/lr_schedules_25.png
results/figures/lr_schedules_100.png
results/figures/lr_schedules_400.png
results/figures/main_comparison_grid.png
results/figures/final_error_bar.png
results/figures/stage_error_bar.png
results/figures/ablation_bar.png
results/figures/correction_coefficients.png
```

单条曲线诊断图：

```text
results/figures/prediction_25_wsd_20000_24000.png
results/figures/prediction_25_wsdld_20000_24000.png
results/figures/prediction_100_wsd_20000_24000.png
results/figures/prediction_100_wsdld_20000_24000.png
results/figures/prediction_400_wsd_20000_24000.png
results/figures/prediction_400_wsdld_20000_24000.png
results/figures/residual_25_wsd_20000_24000.png
results/figures/residual_25_wsdld_20000_24000.png
results/figures/residual_100_wsd_20000_24000.png
results/figures/residual_100_wsdld_20000_24000.png
results/figures/residual_400_wsd_20000_24000.png
results/figures/residual_400_wsdld_20000_24000.png
```

## Task 1: 添加实验划分配置

**文件：**

- 新建：`src/splits.py`
- 测试：`tests/test_features.py`

- [ ] **Step 1: 写入 split 常量**

创建 `src/splits.py`，包含以下公开常量：

```python
SIZES = ["25", "100", "400"]

REPRODUCTION_TRAIN_CURVES = [
    "cosine_24000.csv",
    "constant_24000.csv",
    "wsdcon_9.csv",
]

STRICT_COSINE_TRAIN_CURVES = [
    "cosine_24000.csv",
]

MAIN_TEST_CURVES = [
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
]

AUX_TEST_CURVES = [
    "constant_72000.csv",
    "cosine_72000.csv",
    "wsdcon_3.csv",
    "wsdcon_18.csv",
]

ALL_EVAL_CURVES = MAIN_TEST_CURVES + AUX_TEST_CURVES

PHASES = {
    "warmup_or_start": (0, 2160),
    "post_warmup_stable": (2160, 20000),
    "decay": (20000, 24000),
    "long_horizon": (24000, None),
}
```

- [ ] **Step 2: 添加阶段 mask helper**

在 `src/splits.py` 中继续添加：

```python
import numpy as np


def phase_mask(steps: np.ndarray, phase_name: str) -> np.ndarray:
    start, end = PHASES[phase_name]
    if end is None:
        return steps >= start
    return (steps >= start) & (steps < end)
```

- [ ] **Step 3: 测试阶段 mask**

创建 `tests/test_features.py`：

```python
import numpy as np
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
```

- [ ] **Step 4: 运行测试**

运行：

```powershell
python -m pytest tests/test_features.py -q
```

期望：

```text
2 passed
```

## Task 2: 实现 schedule 特征

**文件：**

- 新建：`src/features.py`
- 修改：`tests/test_features.py`

- [ ] **Step 1: 添加特征函数**

创建 `src/features.py`：

```python
import numpy as np


def cumulative_lr(lrs: np.ndarray) -> np.ndarray:
    return np.cumsum(lrs)


def positive_decay(lrs: np.ndarray) -> np.ndarray:
    decay = np.zeros_like(lrs, dtype=float)
    decay[1:] = np.maximum(lrs[:-1] - lrs[1:], 0.0)
    return decay


def tissue_s2(lrs: np.ndarray, lambda_decay: float) -> np.ndarray:
    decay = positive_decay(lrs)
    momentum = np.zeros_like(lrs, dtype=float)
    for i in range(1, len(lrs)):
        momentum[i] = lambda_decay * momentum[i - 1] + decay[i]
    return np.cumsum(momentum)


def correction_features(lrs: np.ndarray, steps: np.ndarray) -> tuple[np.ndarray, list[str]]:
    s1 = cumulative_lr(lrs)
    decay = positive_decay(lrs)
    decay_cum = np.cumsum(decay)
    eta_max = max(float(np.max(lrs)), 1e-12)
    eta_norm = lrs / eta_max
    s1_sq = np.cumsum(lrs ** 2)
    mem_099 = tissue_s2(lrs, 0.99)
    mem_0995 = tissue_s2(lrs, 0.995)
    mem_0999 = tissue_s2(lrs, 0.999)
    total = max(len(lrs) - 1, 1)
    t_norm = np.arange(len(lrs), dtype=float) / total
    is_decay = (decay_cum > 0).astype(float)

    full = np.column_stack([
        np.ones(len(lrs)),
        np.log(np.maximum(s1, 1e-12)),
        eta_norm,
        s1_sq,
        decay_cum,
        mem_099,
        mem_0995,
        mem_0999,
        t_norm,
        is_decay,
    ])
    names = [
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
    return full[steps], names
```

- [ ] **Step 2: 添加特征测试**

追加到 `tests/test_features.py`：

```python
from src.features import positive_decay, tissue_s2, correction_features


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
```

- [ ] **Step 3: 运行特征测试**

运行：

```powershell
python -m pytest tests/test_features.py -q
```

期望：

```text
5 passed
```

## Task 3: 实现 Tissue baseline

**文件：**

- 新建：`src/annealing_law.py`
- 测试：`tests/test_correction.py`

- [ ] **Step 1: 添加 Tissue predictor**

创建 `src/annealing_law.py`：

```python
import numpy as np
from scipy.optimize import minimize
from src.features import cumulative_lr, tissue_s2
from src.utils import huber_loss


class TissueAnnealingLaw:
    def __init__(self, lambda_decay: float = 0.995):
        self.lambda_decay = lambda_decay
        self.params = None

    def predict_curve(self, lrs: np.ndarray, steps: np.ndarray, params=None) -> np.ndarray:
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("TissueAnnealingLaw must be fitted before prediction.")
        l0, a, alpha, c = params
        s1 = cumulative_lr(lrs)[steps]
        s2 = tissue_s2(lrs, self.lambda_decay)[steps]
        return l0 + a * np.maximum(s1, 1e-12) ** (-alpha) - c * s2

    def fit(self, data: dict, train_curves: list[str]):
        min_loss = min(float(data[name]["loss"].min()) for name in train_curves)
        init = np.array([min_loss - 0.05, 0.5, 0.5, 0.5], dtype=float)

        def objective(params):
            if np.any(params <= 0):
                return 1e9
            total = 0.0
            for name in train_curves:
                pred = self.predict_curve(data[name]["lrs"], data[name]["step"], params=params)
                loss = data[name]["loss"]
                pred = np.maximum(pred, 1e-10)
                residual = np.log(loss) - np.log(pred)
                total += float(huber_loss(residual).sum())
            return total

        result = minimize(
            objective,
            init,
            method="L-BFGS-B",
            bounds=[(1e-8, None), (1e-8, None), (1e-8, 5.0), (1e-8, None)],
            options={"maxiter": 10000, "ftol": 1e-12},
        )
        self.params = result.x
        return self.params, float(result.fun)


def fit_best_tissue(data: dict, train_curves: list[str], lambdas=(0.99, 0.995, 0.999)):
    best = None
    for lambda_decay in lambdas:
        model = TissueAnnealingLaw(lambda_decay=lambda_decay)
        params, loss = model.fit(data, train_curves)
        if best is None or loss < best["loss"]:
            best = {"model": model, "params": params, "loss": loss, "lambda_decay": lambda_decay}
    return best
```

- [ ] **Step 2: 添加 smoke test**

创建 `tests/test_correction.py`：

```python
import numpy as np
from src.annealing_law import fit_best_tissue


def test_fit_best_tissue_on_synthetic_power_curve():
    lrs = np.full(20, 0.1)
    steps = np.arange(1, 20)
    s1 = np.cumsum(lrs)[steps]
    loss = 1.0 + 0.5 * s1 ** -0.5
    data = {"constant.csv": {"lrs": lrs, "step": steps, "loss": loss}}
    best = fit_best_tissue(data, ["constant.csv"], lambdas=(0.99,))
    pred = best["model"].predict_curve(lrs, steps)
    assert np.mean(np.abs(pred - loss)) < 0.02
```

- [ ] **Step 3: 运行测试**

运行：

```powershell
python -m pytest tests/test_correction.py -q
```

期望：

```text
1 passed
```

## Task 4: 添加纯预测 wrapper

**文件：**

- 新建：`src/predictors.py`
- 测试：`tests/test_metrics.py`

- [ ] **Step 1: 添加 MPL 预测函数**

创建 `src/predictors.py`：

```python
import numpy as np


def predict_mpl_curve(lrs: np.ndarray, steps: np.ndarray, params: list[float]) -> np.ndarray:
    l0, a, alpha, b, c, beta, gamma = params
    lr_sum = np.cumsum(lrs)
    lr_gap = np.zeros(len(lrs), dtype=float)
    lr_gap[1:] = np.diff(lrs)
    s1 = lr_sum[steps]
    ld = np.zeros(len(steps), dtype=float)
    for i, step in enumerate(steps):
        if step > 0:
            scaled = lrs[1:step + 1] ** (-gamma) * (lr_sum[step] - lr_sum[:step])
            ld[i] = np.sum(lr_gap[1:step + 1] * (1 - (1 + c * scaled) ** (-beta)))
    return l0 + a * np.maximum(s1, 1e-12) ** (-alpha) + b * ld
```

- [ ] **Step 2: 添加预测形状测试**

创建 `tests/test_metrics.py`：

```python
import numpy as np
from src.predictors import predict_mpl_curve


def test_predict_mpl_curve_returns_one_prediction_per_observed_step():
    lrs = np.full(10, 0.1)
    steps = np.array([1, 5, 9])
    params = [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5]
    pred = predict_mpl_curve(lrs, steps, params)
    assert pred.shape == steps.shape
    assert np.all(np.isfinite(pred))
```

- [ ] **Step 3: 运行 predictor 测试**

运行：

```powershell
python -m pytest tests/test_metrics.py -q
```

期望：

```text
1 passed
```

## Task 5: 实现指标和阶段诊断

**文件：**

- 新建：`src/metrics.py`
- 修改：`tests/test_metrics.py`

- [ ] **Step 1: 添加指标函数**

创建 `src/metrics.py`：

```python
import numpy as np
from sklearn.metrics import r2_score
from src.splits import PHASES, phase_mask


def curve_metrics(loss: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - loss
    rel = np.abs(error) / np.maximum(np.abs(loss), 1e-12)
    auc_loss = np.trapz(loss)
    auc_pred = np.trapz(pred)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "prede": float(np.mean(rel)),
        "worste": float(np.max(rel)),
        "r2": float(r2_score(loss, pred)),
        "final_rel_error": float(abs(pred[-1] - loss[-1]) / max(abs(loss[-1]), 1e-12)),
        "auc_rel_error": float(abs(auc_pred - auc_loss) / max(abs(auc_loss), 1e-12)),
    }


def stage_metrics(steps: np.ndarray, loss: np.ndarray, pred: np.ndarray) -> list[dict]:
    rows = []
    for phase_name, (start, end) in PHASES.items():
        mask = phase_mask(steps, phase_name)
        n_points = int(mask.sum())
        if n_points == 0:
            rows.append({
                "stage": phase_name,
                "start_step": start,
                "end_step": end,
                "n_points": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "prede": np.nan,
                "mean_signed_error": np.nan,
            })
            continue
        error = pred[mask] - loss[mask]
        rel = np.abs(error) / np.maximum(np.abs(loss[mask]), 1e-12)
        rows.append({
            "stage": phase_name,
            "start_step": start,
            "end_step": end,
            "n_points": n_points,
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "prede": float(np.mean(rel)),
            "mean_signed_error": float(np.mean(error)),
        })
    return rows
```

- [ ] **Step 2: 添加指标测试**

追加到 `tests/test_metrics.py`：

```python
from src.metrics import curve_metrics, stage_metrics


def test_curve_metrics_exact_prediction_is_zero_error():
    loss = np.array([3.0, 2.5, 2.0])
    pred = loss.copy()
    metrics = curve_metrics(loss, pred)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["final_rel_error"] == 0.0


def test_stage_metrics_records_nan_for_empty_stage():
    steps = np.array([2160, 20000, 23936])
    loss = np.array([3.0, 2.8, 2.7])
    pred = np.array([3.1, 2.9, 2.6])
    rows = stage_metrics(steps, loss, pred)
    warmup = [row for row in rows if row["stage"] == "warmup_or_start"][0]
    assert warmup["n_points"] == 0
    assert np.isnan(warmup["rmse"])
```

- [ ] **Step 3: 运行指标测试**

运行：

```powershell
python -m pytest tests/test_metrics.py -q
```

期望：

```text
3 passed
```

## Task 6: 实现 decay-aware residual correction

**文件：**

- 新建：`src/correction.py`
- 修改：`tests/test_correction.py`

- [ ] **Step 1: 添加闭式解 ridge regression**

创建 `src/correction.py`：

```python
import numpy as np


class RidgeResidualCorrection:
    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.coef_ = None
        self.feature_names_ = None

    def fit(self, features: np.ndarray, residual: np.ndarray, feature_names: list[str]):
        xtx = features.T @ features
        penalty = self.alpha * np.eye(xtx.shape[0])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(xtx + penalty, features.T @ residual)
        self.feature_names_ = feature_names
        return self

    def predict_residual(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("RidgeResidualCorrection must be fitted before prediction.")
        return features @ self.coef_


def stack_residual_training_rows(data: dict, train_curves: list[str], base_predictions: dict, feature_fn):
    feature_blocks = []
    residual_blocks = []
    feature_names = None
    for name in train_curves:
        features, names = feature_fn(data[name]["lrs"], data[name]["step"])
        if feature_names is None:
            feature_names = names
        residual = data[name]["loss"] - base_predictions[name]
        feature_blocks.append(features)
        residual_blocks.append(residual)
    return np.vstack(feature_blocks), np.concatenate(residual_blocks), feature_names
```

- [ ] **Step 2: 添加 correction 测试**

追加到 `tests/test_correction.py`：

```python
from src.correction import RidgeResidualCorrection


def test_ridge_residual_correction_recovers_linear_residual():
    x = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
    y = 2.0 + 0.5 * np.arange(5, dtype=float)
    model = RidgeResidualCorrection(alpha=0.0).fit(x, y, ["bias", "x"])
    pred = model.predict_residual(x)
    assert np.allclose(pred, y)
```

- [ ] **Step 3: 运行 correction 测试**

运行：

```powershell
python -m pytest tests/test_correction.py -q
```

期望：

```text
2 passed
```

## Task 7: 编写 reproduction 脚本

**文件：**

- 新建：`scripts/run_reproduction.py`
- 新建目录：`results/metrics`、`results/predictions`、`results/figures`

- [ ] **Step 1: 创建脚本和结果目录**

运行：

```powershell
New-Item -ItemType Directory -Force -Path scripts
New-Item -ItemType Directory -Force -Path results/metrics
New-Item -ItemType Directory -Force -Path results/predictions
New-Item -ItemType Directory -Force -Path results/figures
```

期望：上述目录存在。

- [ ] **Step 2: 实现 reproduction 脚本**

创建 `scripts/run_reproduction.py`，完成：

1. 遍历模型规模 `25`、`100`、`400`。
2. 通过 `load_data(FOLDER_PATHS[size])` 读取数据。
3. 在 `REPRODUCTION_TRAIN_CURVES` 上拟合 Tissue baseline。
4. 通过现有 `initialize_params`、`generate_init_params`、`mpl_adam_fit` 拟合 MPL。
5. 在 `ALL_EVAL_CURVES` 上评估两个方法。
6. 写出 `results/metrics/reproduction_metrics.csv`。

run id 使用：

```text
reproduction_paper_split
```

- [ ] **Step 3: 运行 reproduction 脚本**

运行：

```powershell
python scripts/run_reproduction.py
```

期望生成：

```text
results/metrics/reproduction_metrics.csv
```

CSV 行数应为：

```text
3 sizes * 2 methods * 6 eval curves = 36 rows
```

## Task 8: 编写 cross-schedule 脚本

**文件：**

- 新建：`scripts/run_cross_schedule.py`

- [ ] **Step 1: 实现严格 cosine-to-WSD 实验**

创建 `scripts/run_cross_schedule.py`，完成：

1. 遍历模型规模 `25`、`100`、`400`。
2. 在 `STRICT_COSINE_TRAIN_CURVES` 上拟合 Tissue。
3. 在 `STRICT_COSINE_TRAIN_CURVES` 上拟合 MPL。
4. 对 `ALL_EVAL_CURVES` 中每条曲线预测。
5. residual correction 只在 `STRICT_COSINE_TRAIN_CURVES` 上训练。
6. 对所有 eval curve 输出 `base + correction` 预测。
7. 曲线整体指标写入 `results/metrics/cross_schedule_metrics.csv`。
8. 阶段指标写入 `results/metrics/stage_metrics.csv`。
9. 每个 method/size/curve 的预测写到 `results/predictions/`。

Prediction CSV 字段：

```text
step,loss,pred,residual,method,size,curve
```

- [ ] **Step 2: 运行 cross-schedule 实验**

运行：

```powershell
python scripts/run_cross_schedule.py
```

期望生成：

```text
results/metrics/cross_schedule_metrics.csv
results/metrics/stage_metrics.csv
```

主指标表行数应为：

```text
3 sizes * 4 methods * 6 eval curves = 72 rows
```

四个方法为：

```text
tissue
mpl
tissue_plus_ridge
mpl_plus_ridge
```

## Task 9: 编写 ablation 脚本

**文件：**

- 新建：`scripts/run_ablation.py`

- [ ] **Step 1: 定义 feature sets**

使用以下 feature sets：

```text
bias_only: bias
lr_level: bias, eta_norm, t_norm
cumulative: bias, log_s1, s1_sq, t_norm
decay_cum: bias, log_s1, eta_norm, decay_cum, t_norm
decay_memory: bias, log_s1, eta_norm, decay_mem_099, decay_mem_0995, decay_mem_0999, t_norm
full: all features
```

- [ ] **Step 2: 实现 ablation loop**

创建 `scripts/run_ablation.py`，完成：

1. 从 `results/predictions/` 读取 saved base predictions，或重新计算 base predictions。
2. 对每个 feature set 拟合一个 ridge correction。
3. 在 `MAIN_TEST_CURVES` 上评估。
4. 写出 `results/metrics/ablation_metrics.csv`。
5. 对 full feature set 写出 `results/metrics/correction_coefficients.csv`。

- [ ] **Step 3: 运行 ablation**

运行：

```powershell
python scripts/run_ablation.py
```

期望生成：

```text
results/metrics/ablation_metrics.csv
results/metrics/correction_coefficients.csv
```

Ablation 表行数应为：

```text
3 sizes * 2 base methods * 6 feature sets * 2 main test curves = 72 rows
```

## Task 10: 编写绘图脚本

**文件：**

- 新建：`scripts/make_figures.py`

- [ ] **Step 1: 生成 LR schedule 图**

`scripts/make_figures.py` 应读取每个模型规模，并绘制：

```text
cosine_24000
constant_24000
wsd_20000_24000
wsdld_20000_24000
wsdcon_3
wsdcon_18
```

保存：

```text
results/figures/lr_schedules_25.png
results/figures/lr_schedules_100.png
results/figures/lr_schedules_400.png
```

- [ ] **Step 2: 生成 prediction 和 residual 图**

对每个模型规模和每条主 WSD 曲线，绘制：

1. True loss。
2. Tissue prediction。
3. MPL prediction。
4. MPL + ridge prediction。

按照“图表产出”中列出的文件名保存 prediction plots 和 residual plots。

- [ ] **Step 3: 生成汇总图**

读取 metrics CSV，生成：

```text
results/figures/main_comparison_grid.png
results/figures/stage_error_bar.png
results/figures/ablation_bar.png
results/figures/correction_coefficients.png
```

- [ ] **Step 4: 运行绘图脚本**

运行：

```powershell
python scripts/make_figures.py
```

期望：

```text
results/figures/ 下至少有 20 个 PNG 文件
```

## Task 11: 更新 README 保证可复现

**文件：**

- 修改：`README.md`

- [ ] **Step 1: 添加项目说明小节**

添加小节：

```text
Final Project Experiments
```

包含命令：

```powershell
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_reproduction.py
python scripts/run_cross_schedule.py
python scripts/run_ablation.py
python scripts/make_figures.py
```

- [ ] **Step 2: 说明输出文件**

说明：

```text
results/metrics/reproduction_metrics.csv
results/metrics/cross_schedule_metrics.csv
results/metrics/stage_metrics.csv
results/metrics/ablation_metrics.csv
results/figures/
```

- [ ] **Step 3: 运行 README 中的测试命令**

运行：

```powershell
python -m pytest tests -q
```

期望：

```text
All tests pass
```

## Task 12: Slides 组装清单

**文件：**

- 不需要新增代码文件。
- 使用生成的 CSV 和 PNG。

- [ ] **Step 1: 组织叙事**

slides 顺序：

```text
1. Problem: schedule-aware loss-curve prediction
2. Task requirement and data
3. Tissue annealing law baseline
4. Multi-Power Law baseline
5. Experiment protocol: fit cosine, evaluate WSD/WSDLD
6. Main result table
7. Stage-wise error diagnosis
8. Residual correction method
9. Ablation results
10. Limitations and takeaways
11. Division of labor
12. GitHub repository link
```

- [ ] **Step 2: 使用这些图**

使用：

```text
results/figures/lr_schedules_400.png
results/figures/main_comparison_grid.png
results/figures/stage_error_bar.png
results/figures/ablation_bar.png
results/figures/correction_coefficients.png
```

- [ ] **Step 3: 使用这些表**

使用：

```text
results/metrics/cross_schedule_metrics.csv
results/metrics/stage_metrics.csv
results/metrics/ablation_metrics.csv
```

slides 中只保留最关键行：

```text
size=400
test_curve=wsd_20000_24000.csv
test_curve=wsdld_20000_24000.csv
methods=tissue,mpl,mpl_plus_ridge
```

## 验证命令

完整验证：

```powershell
python -m pytest tests -q
python scripts/run_reproduction.py
python scripts/run_cross_schedule.py
python scripts/run_ablation.py
python scripts/make_figures.py
```

最终应生成：

```text
results/metrics/reproduction_metrics.csv
results/metrics/cross_schedule_metrics.csv
results/metrics/stage_metrics.csv
results/metrics/ablation_metrics.csv
results/metrics/correction_coefficients.csv
results/figures/main_comparison_grid.png
results/figures/stage_error_bar.png
results/figures/ablation_bar.png
results/figures/correction_coefficients.png
```

## 自检

需求覆盖：

- Proposal 中的 reproduction 要求由 Task 3、Task 4、Task 7、Task 8 覆盖。
- 误差诊断要求由 Task 5、Task 8、Task 10 覆盖。
- Decay-aware functional correction 由 Task 2、Task 6、Task 8、Task 9 覆盖。
- 指标表和图表产出已经在“指标表产出”“图表产出”以及 Task 8-10 中指定。
- GitHub reproducibility 由 Task 11 覆盖。

占位符检查：

- 没有保留待填写项或未定义文件名。
- 每个输出文件都有明确路径。
- 每个实验都有明确命令和期望产物。

类型一致性：

- `steps`、`loss`、`pred`、`lrs` 在 feature、predictor、metric 模块中均为 NumPy arrays。
- Baseline predictor 暴露 `predict_curve(lrs, steps)` 或 `predict_mpl_curve(lrs, steps, params)`。
- Residual correction 接收 `features`，并为每个观测 step 输出一个 residual。
