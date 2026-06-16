# Final Project Overview：跨学习率调度的 LLM 预训练损失曲线预测

项目题目：Predicting LLM Pretraining Loss Curves across Learning-Rate Schedules

本文档根据 `docs/task2_improvement_plan_zh.html` 和当前已经完成的三个提升方向，重新整理整个 final project 的实现逻辑、实验设计、结果解释和答辩叙事。它的定位不是逐行 API 手册，而是帮助读者快速理解：这个项目为什么这样做、三个方向分别解决什么问题、当前结果支持怎样的结论，以及应该如何复现。

## 1. 项目定位与最终叙事

本项目研究的问题是：如果只观察一种学习率调度下的预训练损失曲线，能否预测另一类学习率调度下的损失曲线，尤其是 WSD 和 WSDLD 这类带有明显学习率衰减阶段的 schedule。

项目不重新训练 LLM，而是使用课程提供的已有 loss curves，在这些曲线之上拟合 loss-curve predictor。核心设置是：

1. 用 `cosine_24000.csv` 作为严格训练曲线。
2. 在 `wsd_20000_24000.csv` 和 `wsdld_20000_24000.csv` 上评估跨 schedule 外推。
3. 同时保留 `cosine_72000.csv`、`constant_72000.csv`、`wsdcon_3.csv`、`wsdcon_18.csv` 作为辅助泛化检查。

Task 2 提升计划最初提出三个方向：

| 方向 | 实现状态 | 当前解释 |
| --- | --- | --- |
| A. FSL-light residual correction | 已实现 | 用少量 FSL-inspired schedule functional 替代 full ridge 特征堆叠；结果显示它比 naive full ridge 略克制，但没有稳定超过 Tissue/MPL baseline。 |
| B. 稳健性和不确定性实验 | 已实现 | 增加 ridge alpha sweep、MPL multi-start、stage sensitivity，帮助解释结果是否受超参数、初值和训练阶段影响。 |
| C. NCPL-style neural surrogate | 已实现 | 增加直接神经 surrogate 作为高风险 baseline；当前误差很大，支持“课程数据规模太小，不宜把神经 surrogate 作为主结论”的判断。 |

因此，本项目最终叙事不应写成“新方法全面优于 baseline”。更稳妥也更有学术价值的结论是：

> 本项目复现并扩展了 Multi-Power Law loss prediction，在严格 cosine-to-WSD/WSDLD 外推设置下系统检验了 residual correction 的可行性。实验发现 Tissue 和 MPL baseline 已经很强；naive full ridge 容易过拟合 cosine residual；FSL-light 和 alpha sweep 说明低复杂度 schedule functional 能缓解一部分过拟合，但不能保证超过 baseline；NCPL-style surrogate 在小数据设置下明显不稳，适合作为 future work 而不是最终方法。

## 2. 数据、规模与实验划分

数据目录由 `src/config.py` 管理，当前使用三个模型规模：

```text
loss_curve_repo/csv_25
loss_curve_repo/csv_100
loss_curve_repo/csv_400
```

每个 CSV 包含：

```text
step, lr, loss
```

其中 `step` 是训练步数，`lr` 是当前学习率，`loss` 是观测训练损失。

`src/splits.py` 统一定义实验划分：

```python
SIZES = ["25", "100", "400"]

STRICT_COSINE_TRAIN_CURVES = [
    "cosine_24000.csv",
]

MAIN_TEST_CURVES = [
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
]
```

阶段划分同样由 `src/splits.py` 管理：

```python
PHASES = {
    "warmup_or_start": (0, 2160),
    "post_warmup_stable": (2160, 20000),
    "decay": (20000, 24000),
    "long_horizon": (24000, None),
}
```

这个阶段划分很重要，因为 WSD/WSDLD 的关键难点并不只是整条曲线 RMSE，而是 stable-to-decay transition 之后的误差结构是否发生系统变化。

## 3. 整体 pipeline

最终实现围绕五条实验线组织：

1. `scripts/run_reproduction.py`：复现 Tissue 和 MPL 在 paper-like split 下的表现。
2. `scripts/run_cross_schedule.py`：主实验，只用 cosine 训练，评估到 WSD/WSDLD 和辅助曲线，并加入 full ridge 与 FSL-light residual correction。
3. `scripts/run_ablation.py`：比较不同 residual feature set，并做 FSL-light ridge alpha sweep。
4. `scripts/run_robustness.py`：做 MPL multi-start uncertainty 和 stage sensitivity 汇总。
5. `scripts/run_ncpl_surrogate.py`：运行 NCPL-style direct neural surrogate。

图表统一由 `scripts/make_figures.py` 生成。它读取 `results/metrics/*.csv` 和 `results/predictions/*.csv`，输出可直接放进报告或 slides 的 PNG。

## 4. 基线模型：MPL 与 Tissue

### 4.1 MPL

MPL 预测器位于 `src/predictors.py`：

```python
predict_mpl_curve(lrs, steps, params)
```

参数形式为：

```python
l0, a, alpha, b, c, beta, gamma = params
```

核心思想是把 loss 写成累计学习率幂律项加上学习率变化项：

```text
loss_pred = l0 + a * S1(t)^(-alpha) + b * LD(t)
```

其中 `S1(t)` 是累计学习率，`LD(t)` 用学习率变化和饱和幂律函数刻画 schedule decay 对 loss 的影响。主实验中 MPL-like 参数通过 `src/experiment_utils.py` 里的 SciPy L-BFGS-B wrapper 在 `cosine_24000.csv` 上重新拟合。

### 4.2 Tissue Annealing Law

Tissue baseline 位于 `src/annealing_law.py`。它使用更简单的 annealing memory：

```text
l0 + a * S1(t)^(-alpha) - c * S2(t)
```

其中 `S2(t)` 来自 `src/features.py` 里的 `tissue_s2()`：

```python
momentum[i] = lambda_decay * momentum[i - 1] + positive_decay[i]
S2 = cumsum(momentum)
```

这表示学习率下降信号会被保留为一段 decay memory。Tissue 的优点是参数少、解释直接；当前主实验中它也是最稳定的 baseline 之一。

## 5. Direction A：FSL-light residual correction

Task 2 提升计划指出，原先的 full ridge correction 容易把一条 cosine train curve 上的 residual shape 学死，导致跨 schedule 外推失败。Direction A 的目标是把 residual correction 从“大而全的特征堆叠”改为更轻量、更有文献动机的 FSL-light。

### 5.1 Full ridge 与 FSL-light 的区别

full ridge 使用 `src/features.py` 中的 `correction_features()`，包含 10 个特征：

```text
bias, log_s1, eta_norm, s1_sq, decay_cum,
decay_mem_099, decay_mem_0995, decay_mem_0999,
t_norm, is_decay
```

FSL-light 使用 `fsl_light_features()`，只保留 5 个特征：

```text
bias, log_tau, eta_norm, decay_conv, is_decay
```

其中：

| 特征 | 含义 |
| --- | --- |
| `bias` | 学习整体残差偏移 |
| `log_tau` | 归一化累计学习率的对数坐标，表示 intrinsic optimization time |
| `eta_norm` | 当前学习率相对峰值 |
| `decay_conv` | 平滑后的学习率衰减记忆 |
| `is_decay` | 是否已经进入 decay 相关状态 |

残差修正由 `src/correction.py` 中的 `RidgeResidualCorrection` 完成。训练目标是：

```text
r(t) = y_true(t) - y_base(t)
r(t) ~= X(t) w
y_pred(t) = y_base(t) + X(t) w
```

其中 bias 项不做 L2 惩罚，其它特征受 ridge alpha 正则化。

### 5.2 主实验结果

`scripts/run_cross_schedule.py` 当前会输出六种方法：

```text
tissue
tissue_plus_ridge
tissue_plus_fsl_light
mpl
mpl_plus_ridge
mpl_plus_fsl_light
```

在 WSD/WSDLD 主测试曲线上的平均结果如下：

| 方法 | 平均 RMSE | 平均 final relative error | 平均 AUC relative error | 解释 |
| --- | ---: | ---: | ---: | --- |
| Tissue | 0.010101 | 0.004618 | 0.002255 | 最稳定的简单 baseline。 |
| Tissue + full ridge | 0.023623 | 0.006110 | 0.006522 | RMSE 明显变差，说明 full ridge 过拟合。 |
| Tissue + FSL-light | 0.022299 | 0.004571 | 0.006112 | final error 接近 Tissue，但整体 RMSE 仍变差。 |
| MPL | 0.010571 | 0.010235 | 0.001878 | 强 baseline，AUC error 最低。 |
| MPL + full ridge | 0.028372 | 0.007102 | 0.007580 | final error 有改善，但整条曲线明显变差。 |
| MPL + FSL-light | 0.028353 | 0.010768 | 0.006129 | 比 full ridge 的 AUC error 略低，但 RMSE 仍高。 |

因此，Direction A 的当前结论是：FSL-light 更适合作为“有约束 residual correction 的检验”，而不是最终胜出的预测器。它支持了提升计划里的核心判断：在只有一条 cosine train curve 的条件下，residual correction 很容易学到不可迁移的形状；简单、可解释的 baseline 反而更可靠。

### 5.3 Ablation 结果

`scripts/run_ablation.py` 比较了 residual feature sets：

```text
bias_only, lr_level, cumulative, decay_cum,
decay_memory, full, fsl_light
```

当前结果显示：

1. 对 Tissue base，`lr_level` 的平均 RMSE 为 0.008984，略优于原始 Tissue 的 0.010101。
2. `cumulative` 和 `decay_cum` 对 Tissue 也有轻微改善，平均 RMSE 约为 0.00988。
3. 对 MPL base，`bias_only` 和 `cumulative` 更稳，复杂 decay 特征与 full/FSL-light 反而变差。
4. 这说明 residual correction 的有效性高度依赖 base predictor 和特征自由度，不能简单认为“更多 schedule 特征一定更好”。

这部分结果很适合放在 report 的 ablation/discussion 中：它给出了比单一 FSL-light 结果更细的解释。

## 6. Direction B：稳健性和不确定性实验

Direction B 的目标不是再提出一个新 predictor，而是让结论更可信：结果是否受 ridge alpha、MPL 初值、训练阶段划分影响？

### 6.1 Ridge alpha sweep

`scripts/run_ablation.py` 会同时输出 `results/metrics/alpha_sweep_metrics.csv`，对 FSL-light 使用：

```text
alpha = 1e-6, 1e-4, 1e-2, 1
```

在 WSD/WSDLD 主测试曲线上的平均 RMSE：

| Base | alpha=1e-6 | alpha=1e-4 | alpha=1e-2 | alpha=1 |
| --- | ---: | ---: | ---: | ---: |
| Tissue + FSL-light | 0.023418 | 0.022299 | 0.022110 | 0.015387 |
| MPL + FSL-light | 0.024346 | 0.028353 | 0.028671 | 0.017510 |

更强正则化能显著降低 FSL-light 的过拟合，但仍没有稳定超过 Tissue/MPL baseline。这个结果强化了一个重要结论：当前限制主要来自训练信息不足，而不仅是某个 alpha 没调好。

### 6.2 MPL multi-start uncertainty

`scripts/run_robustness.py` 使用 deterministic variants around `PARAMS[size]` 作为多个初值，重新拟合 MPL，并输出：

```text
results/metrics/mpl_multistart_metrics.csv
results/metrics/mpl_multistart_summary.csv
```

按模型规模汇总后的 WSD/WSDLD RMSE mean/std：

| Size | RMSE mean | RMSE std |
| --- | ---: | ---: |
| 25M | 0.009677 | 0.003145 |
| 100M | 0.010779 | 0.003253 |
| 400M | 0.014272 | 0.003979 |

这说明 MPL 拟合存在一定初值敏感性，但量级仍接近主实验 baseline。报告中可以把它作为“非凸拟合不确定性”的证据，而不是把某一次 MPL refit 当作唯一结论。

### 6.3 Stage sensitivity

`scripts/run_robustness.py` 还会读取 `stage_metrics.csv`，输出：

```text
results/metrics/stage_sensitivity_metrics.csv
```

关键字段包括：

```text
stable_rmse
decay_rmse
decay_minus_stable_rmse
decay_to_stable_rmse_ratio
```

主测试曲线上的平均结果显示：

1. MPL 的 decay RMSE 明显高于 stable RMSE，平均 ratio 约为 5.70。
2. Tissue 的 decay/stable ratio 约为 1.44，更均衡。
3. full ridge 和 FSL-light 往往会把误差从 decay 段转移到 stable 段，导致整体 RMSE 不一定下降。

这解释了为什么只看 final error 容易误判。某些 residual correction 会改善最后一点，却牺牲整条曲线或稳定阶段的误差。

## 7. Direction C：NCPL-style neural surrogate

Direction C 参考 NCPL-style 思路，尝试训练一个直接从配置和 schedule functional 预测 log loss 的神经 surrogate。它位于：

```text
src/surrogate.py
scripts/run_ncpl_surrogate.py
```

特征包括：

```text
log_model_size, step_norm, log_step,
log_tau, eta_norm, decay_conv, is_decay
```

模型是带 `StandardScaler` 的 `MLPRegressor`，训练目标是 log loss：

```text
features -> log(loss)
prediction = exp(model(features))
```

当前输出：

```text
results/metrics/ncpl_surrogate_metrics.csv
results/metrics/ncpl_surrogate_stage_metrics.csv
results/predictions/ncpl_surrogate_cosine_train_<size>_ncpl_surrogate_<curve>.csv
results/figures/ncpl_surrogate_comparison.png
```

在 WSD/WSDLD 主测试曲线上的平均表现：

| 方法 | 平均 RMSE | 平均 final relative error | 平均 AUC relative error |
| --- | ---: | ---: | ---: |
| NCPL-style surrogate | 1.460152 | 0.044363 | 0.427431 |

这个结果远差于 Tissue/MPL baseline。它不是失败的代码，而是有用的 negative evidence：当前课程数据只有少量 schedule 和三个模型规模，不足以支撑端到端神经 surrogate。报告中建议把 Direction C 放在 future work 或 limitations 中，说明如果有更大规模训练日志，NCPL-style 方法可能更有意义。

## 8. 指标文件与行数

完整运行后，`results/metrics/` 中包含：

| 文件 | 行数 | 含义 |
| --- | ---: | --- |
| `reproduction_metrics.csv` | 36 | paper-like split 下 Tissue/MPL 复现结果。 |
| `cross_schedule_metrics.csv` | 108 | strict cosine train 下六种方法的跨 schedule 总体指标。 |
| `fsl_light_metrics.csv` | 36 | 从主实验中过滤出的 FSL-light rows。 |
| `stage_metrics.csv` | 432 | 六种方法在四个阶段上的 stage-wise metrics。 |
| `ablation_metrics.csv` | 84 | residual feature set 消融。 |
| `alpha_sweep_metrics.csv` | 48 | FSL-light ridge alpha 敏感性。 |
| `mpl_multistart_metrics.csv` | 30 | MPL 多初值逐曲线结果。 |
| `mpl_multistart_summary.csv` | 6 | MPL 多初值 mean/std/min/max 汇总。 |
| `stage_sensitivity_metrics.csv` | 36 | stable 与 decay 阶段差异。 |
| `ncpl_surrogate_metrics.csv` | 18 | Direction C 总体指标。 |
| `ncpl_surrogate_stage_metrics.csv` | 72 | Direction C stage-wise metrics。 |
| `correction_coefficients.csv` | 60 | full ridge 特征系数。 |

总体指标来自 `src/metrics.py`：

| 指标 | 含义 |
| --- | --- |
| `rmse` | 均方根误差，强调较大的点误差 |
| `mae` | 平均绝对误差 |
| `prede` | 平均相对误差 |
| `worste` | 最大相对误差 |
| `r2` | 决定系数 |
| `final_rel_error` | 最后一个 step 的相对误差 |
| `auc_rel_error` | 整条 loss curve 面积的相对误差 |

`results/predictions/` 中的逐 step CSV 包含：

```text
step, loss, pred, residual, method, size, curve
```

注意：prediction CSV 里的 `residual = loss - pred`，而 `curve_metrics()` 内部的 error 是 `pred - loss`。两者符号相反，这是为了让图中 residual 更直观地表示“真实 loss 还比预测高多少”。

## 9. 图表输出

`scripts/make_figures.py` 生成的主要图表包括：

```text
results/figures/lr_schedules_*.png
results/figures/prediction_*.png
results/figures/residual_*.png
results/figures/main_comparison_grid.png
results/figures/final_error_bar.png
results/figures/stage_error_bar.png
results/figures/fsl_light_comparison.png
results/figures/stage_fsl_light_error.png
results/figures/alpha_sweep_sensitivity.png
results/figures/mpl_multistart_uncertainty.png
results/figures/stage_sensitivity.png
results/figures/ncpl_surrogate_comparison.png
results/figures/ablation_bar.png
results/figures/correction_coefficients.png
```

推荐报告中重点使用：

1. `main_comparison_grid.png`：展示 baseline 与 full ridge 的主对比。
2. `fsl_light_comparison.png`：展示 baseline、full ridge、FSL-light 的完整对比。
3. `stage_fsl_light_error.png`：解释 residual correction 为什么可能牺牲 stable 阶段。
4. `alpha_sweep_sensitivity.png`：展示 alpha 增大后过拟合缓解但仍未超过 baseline。
5. `ncpl_surrogate_comparison.png`：作为 Direction C 的 high-risk baseline evidence。

## 10. 测试覆盖

测试位于 `tests/`，覆盖：

```text
test_features.py
test_metrics.py
test_correction.py
test_cross_schedule.py
test_ablation.py
test_robustness.py
test_surrogate.py
test_ncpl_surrogate.py
test_figures.py
test_experiment_utils.py
test_data_loader.py
test_lrs.py
```

主要验证内容：

1. learning-rate feature 和 phase mask 的边界行为。
2. Tissue/MPL/correction 的基本数值性质。
3. cross-schedule、ablation、robustness、NCPL script 的输出字段和行组织。
4. figure helper 的聚合逻辑与保存路径。

运行：

```bash
python -m pytest tests -q
```

## 11. 推荐复现流程

从仓库根目录运行：

```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_reproduction.py
python scripts/run_cross_schedule.py
python scripts/run_ablation.py
python scripts/run_robustness.py
python scripts/run_ncpl_surrogate.py
python scripts/make_figures.py
```

如果只想复现最核心结果，可以运行：

```bash
python scripts/run_cross_schedule.py
python scripts/run_ablation.py
python scripts/run_robustness.py
python scripts/make_figures.py
```

其中 `run_robustness.py` 依赖 `stage_metrics.csv`，所以需要先运行 `run_cross_schedule.py`。

## 12. 如何阅读代码

建议阅读顺序：

1. `src/splits.py`：理解训练曲线、测试曲线和阶段划分。
2. `src/features.py`：理解 cumulative LR、decay signal、full ridge features 和 FSL-light features。
3. `src/annealing_law.py`：理解 Tissue baseline。
4. `src/predictors.py`：理解 MPL curve prediction。
5. `src/correction.py`：理解 ridge residual correction。
6. `src/experiment_utils.py`：理解拟合、CSV 写出、多初值和汇总 helper。
7. `scripts/run_cross_schedule.py`：理解主实验。
8. `scripts/run_ablation.py`：理解 feature ablation 和 alpha sweep。
9. `scripts/run_robustness.py`：理解 Direction B。
10. `src/surrogate.py` 与 `scripts/run_ncpl_surrogate.py`：理解 Direction C。
11. `scripts/make_figures.py`：理解结果如何转成图表。

这个顺序和 report 的叙事顺序基本一致。

## 13. 报告与答辩建议

推荐报告结构：

1. Introduction：说明学习率调度改变会影响 loss curve，跨 schedule 预测有实际意义。
2. Related Work / Reference Review：对应 Task 2 参考文献，解释 Tissue、MPL、FSL、NCPL-style surrogate 的启发。
3. Data and Setup：说明三个 model size、strict cosine train、WSD/WSDLD test 和 phase split。
4. Baselines：介绍 MPL 与 Tissue。
5. Direction A：介绍 full ridge 失败、FSL-light 设计和 ablation。
6. Direction B：介绍 alpha sweep、multi-start uncertainty、stage sensitivity。
7. Direction C：介绍 NCPL-style surrogate，并明确它是 high-risk/future-work baseline。
8. Discussion：强调 baseline 很强、residual correction 信息不足、复杂模型不适合小数据。
9. Limitations：说明不重新训练 LLM、schedule 数量少、MPL refit 与原仓库 AdamW 实现不同。

答辩时可以用下面的顺序讲：

1. cosine、WSD、WSDLD 都是现实 LLM 训练中重要的 schedule。
2. 我们先复现 Tissue 和 MPL，并建立严格 cosine-to-WSD/WSDLD 外推设置。
3. full ridge 尝试修正 residual，但它在整体 RMSE 上明显变差，说明只用一条 cosine train curve 容易过拟合。
4. FSL-light 把 correction 限制到少量 intrinsic-time 和 decay functional，结果显示更克制但仍没有稳定超过 baseline。
5. Direction B 的 alpha sweep 和 stage sensitivity 解释了原因：强正则能缓解过拟合，但 residual correction 可能把误差从 decay 段转移到 stable 段。
6. Direction C 的 NCPL-style surrogate 误差很大，说明当前数据规模不适合端到端神经 surrogate。
7. 最终贡献是一个诚实、可复现的跨 schedule evaluation pipeline，以及关于 residual correction 在小数据设置下何时失效的系统诊断。

## 14. 已知限制

1. 本项目不训练新的 LLM，只对已有 loss curves 做拟合和外推评估。
2. 主实验只用一条 `cosine_24000.csv` 训练 residual correction，训练信息非常有限。
3. `run_cross_schedule.py` 中的 MPL-like fitting 使用 SciPy L-BFGS-B wrapper，不完全等同于原仓库 PyTorch AdamW 训练流程。
4. full ridge 和 FSL-light 的失败不应被解释为实现 bug，而应被解释为 cosine-only residual learning 的泛化风险。
5. NCPL-style surrogate 当前数据量不足，结果只能作为 future work 的动机，不适合作为最终方法。
6. 当前图表面向报告展示，不是交互式分析工具；深入分析应直接读取 `results/metrics/*.csv`。

## 15. 一句话总结

本 final project 把原 Multi-Power Law 仓库扩展为一个完整的跨学习率调度损失曲线预测评估平台：它复现 Tissue/MPL baseline，实现 FSL-light residual correction、稳健性/不确定性实验和 NCPL-style surrogate，并用当前结果说明，在课程数据规模下，简单可解释 baseline 比复杂 residual 或神经 surrogate 更可靠，而 residual correction 的主要价值是揭示跨 schedule 外推中的过拟合与阶段性误差结构。
