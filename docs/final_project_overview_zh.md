# Final Project 全项目说明文档

项目题目：Predicting LLM Pretraining Loss Curves across Learning-Rate Schedules

本文档用于解释本 final project 的完整实现思路：研究问题是什么，核心模型为什么这样设计，代码模块如何对应数学原理，实验脚本如何运行，以及指标、表格和图表分别说明什么。

## 1. 项目目标

本项目研究的问题是：如果只观察某些学习率调度下的预训练损失曲线，能否预测另一类学习率调度下的损失曲线，尤其是 WSD 和 WSDLD 这类带有学习率衰减阶段的 schedule。

更具体地说，仓库中已经提供了不同模型规模、不同学习率调度下的损失曲线。我们不训练新的 LLM，而是在这些已有曲线之上拟合 loss-curve predictor。项目重点不是重新跑大模型预训练，而是比较不同损失曲线预测公式在跨 learning-rate schedule 时的外推能力。

本项目实现了三类预测器：

1. MPL, Multi-Power Law, 对应原仓库中的主要 loss prediction 形式。
2. Tissue Annealing Law, 一个较简单、可解释的 annealing baseline。
3. Base predictor + residual correction, 在 MPL 或 Tissue 的基础上，用学习率相关特征修正残差；其中 full ridge 作为过拟合诊断，FSL-light 作为轻量、可解释的最终对比方案。

核心实验问题是：

1. MPL 是否能从 cosine schedule 外推到 WSD/WSDLD schedule。
2. Tissue 这种 decay-memory 形式是否能捕捉学习率衰减带来的误差结构。
3. 用 residual correction 加入学习率衰减特征后，是否能改善跨 schedule 预测。
4. full ridge 的失败是否揭示了 cosine-only 训练下的过拟合风险，FSL-light 是否能提供更稳健的轻量替代。
5. 哪些特征对残差修正最有用。

## 2. 数据与实验划分

数据由 `src.config.FOLDER_PATHS` 指定，当前路径为：

```text
loss_curve_repo/csv_25
loss_curve_repo/csv_100
loss_curve_repo/csv_400
```

每个 CSV 包含三列：

```text
step, lr, loss
```

其中 `step` 是训练步数索引，`lr` 是该步学习率，`loss` 是观测到的训练损失。

实验规模由 `src/splits.py` 统一管理：

```python
SIZES = ["25", "100", "400"]
```

即 25M、100M、400M 三个模型规模。

### 2.1 复现实验划分

`run_reproduction.py` 使用 `REPRODUCTION_TRAIN_CURVES` 作为训练曲线：

```python
REPRODUCTION_TRAIN_CURVES = [
    "cosine_24000.csv",
    "constant_24000.csv",
    "wsdcon_9.csv",
]
```

评估曲线为：

```python
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
```

因此复现实验会在 6 条评估曲线上比较 Tissue 和 MPL。

### 2.2 主实验划分

主实验更严格，只用 cosine 训练：

```python
STRICT_COSINE_TRAIN_CURVES = [
    "cosine_24000.csv",
]
```

然后评估到 WSD、WSDLD 和辅助曲线。这个设置对应项目真正关心的跨 schedule 外推：模型只在 cosine loss curve 上拟合，然后看它面对学习率衰减 schedule 时会不会系统性偏差。

### 2.3 阶段划分

`src/splits.py` 中定义了训练过程的四个阶段：

```python
PHASES = {
    "warmup_or_start": (0, 2160),
    "post_warmup_stable": (2160, 20000),
    "decay": (20000, 24000),
    "long_horizon": (24000, None),
}
```

`phase_mask()` 使用半开区间：

```python
(steps >= start) & (steps < end)
```

如果 `end is None`，则使用：

```python
steps >= start
```

这个阶段划分用于 `stage_metrics.csv`。它帮助我们判断误差主要出现在 warmup、稳定训练段、decay 段，还是 24000 step 之后的 long-horizon 外推段。

## 3. MPL 预测器

MPL 的预测入口是 `src/predictors.py` 中的：

```python
predict_mpl_curve(lrs, steps, params)
```

参数形式为：

```python
l0, a, alpha, b, c, beta, gamma = params
```

代码中的主项是 cumulative learning-rate power law：

```python
lr_sum = np.cumsum(lrs)
s1 = lr_sum[steps]
l0 + a * np.maximum(s1, 1e-12) ** (-alpha)
```

这里的 `s1` 可以理解为到当前 step 为止累计用掉的 learning-rate budget。随着训练推进，`s1` 增大，`s1 ** (-alpha)` 逐渐下降，因此损失预测值也随训练逐渐下降。

MPL 还包含学习率变化项 `ld`。代码中计算方式是：

```python
lr_gap[1:] = np.diff(lrs)
scaled = lrs[1 : step + 1] ** (-gamma) * (lr_sum[step] - lr_sum[:step])
ld[i] = np.sum(
    lr_gap[1 : step + 1] * (1 - (1 + c * scaled) ** (-beta))
)
```

整体预测为：

```python
loss_pred = l0 + a * s1 ** (-alpha) + b * ld
```

直观解释：

1. `l0` 是不可约损失或最终损失下界。
2. `a * s1 ** (-alpha)` 描述随着累计训练量增加，loss 按 power law 下降。
3. `ld` 用学习率变化 `lr_gap` 和历史累计学习率差值建模 schedule 改变造成的影响。
4. `b, c, beta, gamma` 控制学习率变化项的幅度、饱和形式和时间尺度。

在 `scripts/run_reproduction.py` 中，MPL 默认使用 `src.config.PARAMS` 里的预计算参数：

```python
mpl_params = PARAMS[size]
protocol = "paper_like_train_split_precomputed_mpl"
```

这样做的目的，是让默认复现实验可以快速检查完整 pipeline。如果希望重新拟合 MPL-like 参数，可以使用：

```bash
python scripts/run_reproduction.py --fit-mpl --mpl-maxiter 300
```

需要注意：当前 project scripts 中的 refit 使用的是 `src/experiment_utils.py` 的 SciPy L-BFGS-B wrapper，而不是原仓库 `fitting.py` 中的 PyTorch AdamW 训练流程。这是为了让 final project 实验在普通 NumPy/SciPy 环境中更稳定地复现。

## 4. Tissue Annealing Law

Tissue baseline 在 `src/annealing_law.py` 中实现。核心类是：

```python
class TissueAnnealingLaw:
    def __init__(self, lambda_decay=0.995):
        ...
```

它的预测公式是：

```python
l0 + a * s1 ** (-alpha) - c * s2
```

其中 `s1` 仍然是累计学习率：

```python
s1 = cumulative_lr(lrs)[steps]
```

`s2` 是学习率衰减记忆项：

```python
s2 = tissue_s2(lrs, self.lambda_decay)[steps]
```

`tissue_s2()` 在 `src/features.py` 中定义：

```python
def positive_decay(lrs):
    decay = np.zeros_like(lrs, dtype=float)
    decay[1:] = np.maximum(lrs[:-1] - lrs[1:], 0.0)
    return decay

def tissue_s2(lrs, lambda_decay):
    decay = positive_decay(lrs)
    momentum = np.zeros_like(lrs, dtype=float)
    for i in range(1, len(lrs)):
        momentum[i] = lambda_decay * momentum[i - 1] + decay[i]
    return np.cumsum(momentum)
```

这段代码的含义是：

1. 只把学习率下降视为 decay signal。
2. 如果学习率上升或保持不变，`positive_decay` 为 0。
3. `momentum[i] = lambda_decay * momentum[i - 1] + decay[i]` 表示 decay 影响会随时间保留一段记忆。
4. `np.cumsum(momentum)` 表示把 decay memory 累积为一个持续影响 loss 的状态量。

Tissue 的 `fit()` 使用 Huber loss：

```python
residual = np.log(loss) - np.log(pred)
total += huber(0.001, residual).sum()
```

使用 log residual 的原因是，不同训练阶段 loss 数值尺度不同。直接在原始 loss 上拟合，后期小误差和前期大误差的权重会不均衡。log residual 更接近相对误差意义。

`fit_best_tissue()` 会尝试三个 decay memory 参数：

```python
lambdas=(0.99, 0.995, 0.999)
```

然后选 Huber objective 最小的那个模型。

## 5. Residual Correction

Residual correction 的核心思想是：先让 base predictor 做主要预测，再学习它在训练曲线上的系统误差。

如果 base predictor 的预测是：

```text
y_base(t)
```

真实 loss 是：

```text
y_true(t)
```

那么训练残差为：

```text
r(t) = y_true(t) - y_base(t)
```

项目中用 ridge regression 学习：

```text
r(t) ~= X(t) w
```

最终预测为：

```text
y_pred(t) = y_base(t) + X(t) w
```

实现位于 `src/correction.py`：

```python
class RidgeResidualCorrection:
    def fit(self, features, residual, feature_names):
        xtx = features.T @ features
        penalty = self.alpha * np.eye(xtx.shape[0])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(xtx + penalty, features.T @ residual)
```

这对应 ridge closed-form solution：

```text
w = (X^T X + alpha I)^(-1) X^T r
```

但代码中有一个细节：

```python
penalty[0, 0] = 0.0
```

这表示 bias 项不做 L2 惩罚。这样模型可以自由学习整体偏移，但其它特征系数会被正则化，避免在只有一条 cosine train curve 时过拟合。

需要特别说明：本项目保留 full ridge 结果，是为了把它作为过拟合诊断，而不是把失败结果视为代码错误。full ridge 在只有一条 cosine train curve 的主实验设定下拥有较丰富的特征空间，容易学到不能迁移到 WSD/WSDLD 的残差模式。FSL-light 则是更轻量的 residual correction 替代方案，它用更少的 schedule functional 降低自由度，让修正项更容易解释，也更适合 final project 的跨 schedule 对比。

训练数据由：

```python
stack_residual_training_rows(data, train_curves, base_predictions, feature_fn)
```

统一堆叠。它对每条训练曲线计算：

```python
features, names = feature_fn(lrs, step)
residual = loss - base_predictions[name]
```

然后把所有训练曲线的 `features` 和 `residual` 拼成一个大矩阵。

## 6. Residual Correction 特征

残差修正的特征由 `src/features.py` 中的 `correction_features()` 产生：

```python
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
```

特征名为：

```python
[
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
```

各特征含义如下：

| 特征 | 代码来源 | 作用 |
| --- | --- | --- |
| `bias` | `np.ones` | 学习整体残差偏移 |
| `log_s1` | `log(cumulative_lr)` | 表示训练进度的 power-law 坐标 |
| `eta_norm` | `lrs / max(lrs)` | 当前学习率相对峰值 |
| `s1_sq` | `cumsum(lrs ** 2)` | 累计二阶学习率强度 |
| `decay_cum` | `cumsum(positive_decay)` | 总学习率衰减量 |
| `decay_mem_099` | `tissue_s2(lrs, 0.99)` | 短一些的 decay memory |
| `decay_mem_0995` | `tissue_s2(lrs, 0.995)` | 中等 decay memory |
| `decay_mem_0999` | `tissue_s2(lrs, 0.999)` | 长一些的 decay memory |
| `t_norm` | `arange(n) / (n - 1)` | 归一化训练时间 |
| `is_decay` | `decay_cum > 0` | 是否已经进入 decay 相关状态 |

这些特征的设计意图是把 schedule shape 显式提供给 residual model。MPL 和 Tissue 已经各自包含一定的 schedule 信息，但在只用 cosine 拟合时，模型可能无法准确预测 WSD/WSDLD 在 decay 段的偏差。Residual correction 用这些特征补充学习率 level、累计训练量和 decay memory。

FSL-light 使用 `src/features.py` 中的 `fsl_light_features()`。相比 full ridge，它只保留少量轻量 schedule functional，例如：

| FSL-light 特征 | 含义 |
| --- | --- |
| `bias` | 学习整体残差偏移 |
| `log_tau` | 归一化训练进度的对数坐标 |
| `eta_norm` | 当前学习率相对峰值 |
| `decay_conv` | 平滑后的学习率衰减信号 |
| `is_decay` | 是否已经进入学习率衰减相关状态 |

因此，FSL-light 不是要替代 base predictor 的主要物理形状，而是只对 WSD/WSDLD 中最关键的 schedule 状态做残差修正。

## 7. 指标设计

总体曲线指标由 `src/metrics.py` 中的 `curve_metrics()` 计算：

```python
error = pred - loss
rel = abs(error) / max(abs(loss), 1e-12)
```

输出字段包括：

| 指标 | 代码字段 | 含义 |
| --- | --- | --- |
| RMSE | `rmse` | 均方根误差，强调较大的点误差 |
| MAE | `mae` | 平均绝对误差 |
| PREDE | `prede` | 平均相对误差 |
| WorstE | `worste` | 最大相对误差 |
| R2 | `r2` | 决定系数，衡量整体拟合程度 |
| Final Relative Error | `final_rel_error` | 最后一个 step 的相对误差 |
| AUC Relative Error | `auc_rel_error` | 整条 loss curve 面积的相对误差 |

其中 AUC 通过 `np.trapz` 或 `np.trapezoid` 计算：

```python
auc_loss = _trapz(loss)
auc_pred = _trapz(pred)
```

阶段指标由 `stage_metrics()` 计算。它会遍历 `PHASES`，在每个阶段内计算：

```python
rmse
mae
prede
mean_signed_error
```

`mean_signed_error` 特别有用，因为它保留误差方向：

1. 正值表示预测 loss 偏高。
2. 负值表示预测 loss 偏低。

这能帮助分析模型是在 decay 阶段过于乐观，还是过于保守。

## 8. 实验脚本说明

### 8.1 `scripts/run_reproduction.py`

用途：跑 paper-like split 的复现实验。

主要流程：

1. 遍历 `SIZES = ["25", "100", "400"]`。
2. 加载每个 size 的所有曲线。
3. 使用 `REPRODUCTION_TRAIN_CURVES` 拟合 Tissue。
4. MPL 默认读取 `PARAMS[size]`，或者在 `--fit-mpl` 时调用 SciPy optimizer。
5. 在 `ALL_EVAL_CURVES` 上评估 Tissue 和 MPL。
6. 写出 `results/metrics/reproduction_metrics.csv`。

输出行数：

```text
3 sizes * 2 methods * 6 eval curves = 36 rows
```

默认运行：

```bash
python scripts/run_reproduction.py
```

重新拟合 MPL-like 参数：

```bash
python scripts/run_reproduction.py --fit-mpl --mpl-maxiter 300
```

### 8.2 `scripts/run_cross_schedule.py`

用途：主实验，只用 cosine schedule 训练，评估到 WSD/WSDLD 和辅助曲线。

主要流程：

1. 遍历三个 size。
2. 用 `STRICT_COSINE_TRAIN_CURVES` 拟合 Tissue。
3. 用 `fit_mpl_scipy()` 拟合 MPL-like 参数。
4. 对 Tissue 和 MPL 分别训练 full ridge residual correction 和 FSL-light residual correction。
5. 得到六种 method：

```text
tissue
tissue_plus_ridge
tissue_plus_fsl_light
mpl
mpl_plus_ridge
mpl_plus_fsl_light
```

6. 在 6 条评估曲线上输出总体指标。
7. 在 4 个 phase 上输出阶段指标。
8. 为每个预测结果写出逐 step prediction CSV。

输出：

```text
results/metrics/cross_schedule_metrics.csv
results/metrics/fsl_light_metrics.csv
results/metrics/stage_metrics.csv
results/predictions/*.csv
```

行数：

```text
cross_schedule_metrics.csv:
3 sizes * 6 methods * 6 eval curves = 108 rows

fsl_light_metrics.csv:
3 sizes * 2 FSL-light methods * 6 eval curves = 36 rows

stage_metrics.csv:
3 sizes * 6 methods * 6 eval curves * 4 stages = 432 rows

predictions:
3 sizes * 6 methods * 6 eval curves = 108 CSV files
```

### 8.3 `scripts/run_ablation.py`

用途：比较 residual correction 的不同特征组合。

消融特征集定义在 `FEATURE_SETS`：

```python
FEATURE_SETS = {
    "bias_only": ["bias"],
    "lr_level": ["bias", "eta_norm", "t_norm"],
    "cumulative": ["bias", "log_s1", "s1_sq", "t_norm"],
    "decay_cum": ["bias", "log_s1", "eta_norm", "decay_cum", "t_norm"],
    "decay_memory": [
        "bias",
        "log_s1",
        "eta_norm",
        "decay_mem_099",
        "decay_mem_0995",
        "decay_mem_0999",
        "t_norm",
    ],
    "full": [
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
    ],
    "fsl_light": "fsl_light",
}
```

它只在主要 WSD/WSDLD test curves 上评估：

```python
MAIN_TEST_CURVES = [
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
]
```

输出：

```text
results/metrics/ablation_metrics.csv
results/metrics/alpha_sweep_metrics.csv
results/metrics/correction_coefficients.csv
```

行数：

```text
ablation_metrics.csv:
3 sizes * 2 base methods * 7 feature sets * 2 main curves = 84 rows

alpha_sweep_metrics.csv:
3 sizes * 2 base methods * 4 alpha values * 2 main curves = 48 rows

correction_coefficients.csv:
3 sizes * 2 base methods * 10 full features = 60 rows
```

### 8.4 `scripts/make_figures.py`

用途：把 metrics 和 predictions 转为可用于报告或答辩展示的图。

主要输出：

```text
results/figures/lr_schedules_25.png
results/figures/lr_schedules_100.png
results/figures/lr_schedules_400.png
results/figures/prediction_*_wsd_20000_24000.png
results/figures/prediction_*_wsdld_20000_24000.png
results/figures/residual_*_wsd_20000_24000.png
results/figures/residual_*_wsdld_20000_24000.png
results/figures/main_comparison_grid.png
results/figures/final_error_bar.png
results/figures/stage_error_bar.png
results/figures/fsl_light_comparison.png
results/figures/stage_fsl_light_error.png
results/figures/ablation_bar.png
results/figures/correction_coefficients.png
```

当前主要输出这些 PNG；如果 prediction/residual 逐曲线诊断图存在，则数量会随方法和曲线增加。

其中 `stage_error_bar.png` 使用：

```python
summarize_stage_errors(rows)
```

按以下维度汇总：

```text
3 sizes * 3 methods * 2 stages = 18 bars
```

这里的 3 methods 是：

```text
tissue
mpl
mpl_plus_ridge
```

2 stages 是：

```text
post_warmup_stable
decay
```

图中每个 bar 是对 WSD 和 WSDLD 两条主测试曲线的 RMSE 平均值。该旧图保留原始对照范围，避免和 FSL-light 图完全重复。
新增的 `fsl_light_comparison.png` 直接比较 baseline、full ridge 和 FSL-light 六个方法在 WSD/WSDLD 上的总体 RMSE；`stage_fsl_light_error.png` 比较这六个方法在 stable/decay 阶段的 stage RMSE，用于判断 FSL-light 是否主要改善学习率衰减阶段。

## 9. 结果文件结构

完整运行后，结果目录如下：

```text
results/
├── metrics/
│   ├── reproduction_metrics.csv
│   ├── cross_schedule_metrics.csv
│   ├── stage_metrics.csv
│   ├── fsl_light_metrics.csv
│   ├── alpha_sweep_metrics.csv
│   ├── ablation_metrics.csv
│   └── correction_coefficients.csv
├── predictions/
│   └── cross_schedule_cosine_train_{size}_{method}_{curve}.csv
└── figures/
    ├── lr_schedules_*.png
    ├── prediction_*.png
    ├── residual_*.png
    ├── main_comparison_grid.png
    ├── final_error_bar.png
    ├── stage_error_bar.png
    ├── fsl_light_comparison.png
    ├── stage_fsl_light_error.png
    ├── ablation_bar.png
    └── correction_coefficients.png
```

`predictions` 中每个 CSV 包含：

```text
step, loss, pred, residual, method, size, curve
```

其中：

```text
residual = loss - pred
```

这个方向和 `curve_metrics()` 里的 `error = pred - loss` 不同。这样做是为了图像诊断时更直观地表达“真实值还比预测值高多少”。文档和图表中需要注意这两个符号约定。

## 10. 测试说明

测试位于 `tests/`：

```text
test_features.py
test_metrics.py
test_correction.py
test_figures.py
test_data_loader.py
test_lrs.py
```

覆盖内容：

1. `test_features.py` 检查 cumulative LR、decay signal、phase mask 边界。
2. `test_metrics.py` 检查 MPL 预测、曲线指标、阶段指标。
3. `test_correction.py` 检查 Tissue 拟合、ridge residual correction、训练样本堆叠。
4. `test_figures.py` 检查 stage error 图的汇总逻辑覆盖全部 size。
5. `test_data_loader.py` 和 `test_lrs.py` 保留原仓库基础功能测试，并使用 pytest 临时目录保存图像，避免污染项目根目录。

运行：

```bash
python -m pytest tests -q
```

该命令应全部通过；具体测试数量以当前命令输出为准。

## 11. 推荐复现流程

从仓库根目录运行：

```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_reproduction.py
python scripts/run_cross_schedule.py
python scripts/run_ablation.py
python scripts/make_figures.py
```

如果要更严格地重新拟合 reproduction 中的 MPL-like 参数：

```bash
python scripts/run_reproduction.py --fit-mpl --mpl-maxiter 300
```

如果运行时间可接受，可以继续增大：

```bash
python scripts/run_reproduction.py --fit-mpl --mpl-maxiter 1000
```

主实验和消融实验默认会调用 SciPy optimizer，因此耗时比只读预计算参数更长。实际运行时，`run_cross_schedule.py` 和 `run_ablation.py` 可能各需要一两分钟，取决于机器性能。

## 12. 如何阅读代码

建议按下面顺序阅读：

1. `src/splits.py`：先理解数据集、训练曲线、测试曲线和阶段划分。
2. `src/features.py`：理解累计学习率、正向衰减、decay memory 和 residual correction 特征。
3. `src/predictors.py`：理解 MPL 预测公式如何从学习率序列得到 loss prediction。
4. `src/annealing_law.py`：理解 Tissue baseline 如何拟合和预测。
5. `src/correction.py`：理解 ridge residual correction 如何训练。
6. `src/metrics.py`：理解所有评价指标。
7. `src/experiment_utils.py`：理解数据加载、MPL-like fitting、CSV 写出等公共工具。
8. `scripts/run_cross_schedule.py`：理解主实验完整流程。
9. `scripts/run_ablation.py`：理解特征消融。
10. `scripts/make_figures.py`：理解指标如何转成报告图。

这个阅读顺序从实验设定到模型，再到评估和图表，和 final report 的叙述顺序基本一致。

## 13. 报告写作建议

如果要把本项目写成 final report，可以采用下面结构：

1. Introduction：说明学习率调度改变会影响 loss curve，跨 schedule 预测有实践意义。
2. Data and Setup：说明三种 model size、训练曲线、测试曲线、阶段划分。
3. Methods：介绍 MPL、Tissue Annealing Law、Ridge residual correction。
4. Metrics：介绍 RMSE、PREDE、final relative error、stage-wise error。
5. Results：引用 `cross_schedule_metrics.csv`、`stage_metrics.csv` 和主要图表。
6. Ablation：引用 `ablation_metrics.csv` 和 `correction_coefficients.csv`。
7. Discussion：讨论 decay memory 特征是否改善 WSD/WSDLD 外推，说明 full ridge 失败是过拟合诊断，FSL-light 是轻量、可解释的 residual correction，并指出默认 MPL reproduction 与 fresh refit 的区别。
8. Limitations：说明本项目不重新训练 LLM，只使用已有 loss curves；训练集规模较小，ridge correction 可能依赖当前数据分布。

## 14. 已知限制

1. 本项目不训练新的 LLM，只对已有 loss curves 进行曲线拟合和外推评估。
2. 默认 `run_reproduction.py` 使用 `src.config.PARAMS` 中的预计算 MPL 参数，不是 fresh MPL training。
3. `--fit-mpl` 使用 SciPy L-BFGS-B 拟合 MPL-like 参数，与原仓库 PyTorch AdamW 实现不是完全同一个优化过程。
4. 主实验只用一条 cosine curve 训练 residual correction，因此 ridge regularization 很重要；full ridge 的失败应作为过拟合诊断理解。
5. 当前图表偏向报告展示，不是交互式分析工具。如果要做深入分析，可以直接读取 `results/metrics/*.csv`。

## 15. 一句话总结

本 final project 把原仓库的 Multi-Power Law loss prediction 扩展成一个完整的跨学习率调度评估 pipeline：用 cosine schedule 训练 predictor，测试到 WSD/WSDLD schedule，通过 Tissue decay memory、full ridge 过拟合诊断和 FSL-light 轻量残差修正分析学习率衰减阶段的系统误差，并输出可复现的指标表、逐步预测表和报告图表。
