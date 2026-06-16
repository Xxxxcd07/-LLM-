# Multi-Power Law 仓库

本仓库提供了一个使用 Multi-Power Law（MPL，多幂律）模型来拟合、预测和优化学习率（Learning Rate, LR）调度的框架。它面向机器学习优化领域的研究者和实践者，可用于分析训练损失动态，并为大语言模型推导优化后的学习率调度。本工作支持对大语言模型高效训练策略的研究。更多细节请参阅我们的论文：[arXiv:2503.12811](https://arxiv.org/abs/2503.12811)。

## 结果

下表展示了针对 25M、100M 和 400M 参数模型的数据集拟合 MPL 模型后得到的更新版评估指标和最佳参数。注意，由于重新运行所有实验的计算成本较高，本文关联论文中的结果可能与这里略有不同。

### 公式形式

Multi-Power Law（MPL）模型公式如下：

$$
L(t) = L_0 + A \cdot (S_1(t) + S_W)^{-\alpha} - LD(t)
$$

其中：

- $L(t)$：第 $t$ 步的预测损失。
- $S_1(t) = \sum_{\tau=1}^{t} \eta_{\tau}$：截至第 $t$ 步的学习率累计和。
- $S_W$：预热阶段的累计学习率（固定偏移量）。
- $LD(t) = B \sum_{k=1}^{t} (\eta_{k-1} - \eta_k) \cdot G(\eta_k^{-\gamma} S_k(t))$：损失下降项。
- $S_k(t) = \sum_{\tau=k}^{t} \eta_{\tau}$：从第 $k$ 步到第 $t$ 步的局部累计学习率。
- $G(x) = 1 - (C x + 1)^{-\beta}$：作为非线性变换的幂函数。

### 评估指标

| 模型 | $R^2$ | MAE | RMSE | PredE | WorstE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25M | 0.9988 | 0.00376 | 0.00465 | 0.00110 | 0.00409 |
| 100M | 0.9983 | 0.00435 | 0.00592 | 0.00142 | 0.00583 |
| 400M | 0.9978 | 0.00484 | 0.00730 | 0.00168 | 0.00995 |

- **$R^2$**：决定系数，用于衡量拟合优度。
- **MAE**：平均绝对误差，即平均绝对预测误差。
- **RMSE**：均方根误差，即残差的标准差。
- **PredE**：平均相对预测误差。
- **WorstE**：最大相对预测误差。

### 最佳参数

| 模型 | $L_0$ | $A$ | $\alpha$ | $B$ | $C$ | $\beta$ | $\gamma$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25M | 3.040 | 0.525 | 0.508 | 363.788 | 2.066 | 0.583 | 0.641 |
| 100M | 2.651 | 0.601 | 0.453 | 437.946 | 2.132 | 0.598 | 0.655 |
| 400M | 2.375 | 0.654 | 0.429 | 523.425 | 2.025 | 0.594 | 0.635 |

- **参数**：MPL 模型的系数，经过优化以最小化 Huber 损失。

## 功能特性

- **学习率调度器**：支持 cosine、constant、two-stage、WSD 和 WSDLD 调度。
- **优化**：使用拟合后的 MPL 模型，在非递增约束下推导优化后的学习率调度。
- **评估**：生成指标（如 MSE、$R^2$、Huber loss）以及预测损失与真实损失的对比可视化。
- **测试**：包含单元测试，以保证核心组件的可靠性。

## 安装

### 前置条件

- Python 3.8 或更高版本
- Git

### 设置

1. 克隆仓库：

   ```bash
   git clone https://github.com/thu-yao-01-luo/MultiPowerLaw.git
   cd MultiPowerLaw
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

   所需包包括：`numpy`、`torch`、`scipy`、`matplotlib`、`tqdm`、`sklearn`。

3. 可选：设置虚拟环境：

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 上使用：venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 使用方法

### 运行主脚本

`main.py` 脚本会执行完整流程：数据加载、模型拟合、评估和学习率调度优化。运行方式如下：

```bash
python -u main.py --folder_path 400
```

- `--folder_path` 或 `-f`：模型尺寸（`25`、`100` 或 `400`）。默认值：`400`。

如果只想进行优化，并使用 `config.py` 中的预计算参数：

```bash
python main.py --opt_only --folder_path 400
```

- `--opt_only` 或 `-o`：单独运行优化。

为了便于快速上手，我们提供了 `run_all.sh`，可按模型尺寸依次运行测试和主脚本：

```bash
bash run_all.sh
```

**输出内容**：

- **拟合模型评估**：图像保存在 `./<model_size>M/fit/` 中，例如 `./400M/fit/cosine_24000_mplfit.png`。
- **优化后的学习率调度**：保存为 `./optimized_schedules/<model_size>.npy`，并绘制为 `./optimized_schedules/<model_size>.png`。
- **日志**：训练进度、指标和优化细节保存在 `logs/<model_size>.log` 中。

### 运行测试

单元测试位于 `tests/` 目录中。从根目录执行：

```bash
python -m pytest tests -q
```

- pytest 套件会覆盖学习率调度、数据加载、指标、残差修正、实验配置和图表 helper。

## 项目结构

```text
MultiPowerLaw/
├── src/                 # 核心源代码
│   ├── __init__.py      # 包标记
│   ├── config.py        # 常量和配置
│   ├── data_loader.py   # 数据加载和预处理
│   ├── lr_schedules.py # 学习率调度器实现
│   ├── models.py        # MPL 和 MultiPower 模型
│   ├── fitting.py       # 模型拟合逻辑
│   ├── evaluation.py    # 评估和绘图
│   ├── optimization.py  # 学习率调度优化
│   └── utils.py         # 工具函数
├── tests/               # 单元测试
│   ├── __init__.py      # 包标记
│   ├── test_lrs.py
│   └── test_data_loader.py
├── logs/                # 日志文件
├── main.py              # 入口点
├── requirements.txt     # 依赖项
└── README.md            # 文档
```

### 关键组件

- **`config.py`**：定义数据集、路径（如 `OPT_PATH`）和预计算参数。
- **`lr_schedules.py`**：实现训练数据中使用的学习率调度。
- **`models.py`**：包含 `MPL`（核心模型）和 `MultiPower`（已弃用）。
- **`fitting.py`**：使用 AdamW 和早停机制将 MPL 拟合到训练数据。
- **`optimization.py`**：使用拟合后的 MPL 模型优化学习率调度。
- **`evaluation.py`**：提供指标和可视化。

## 数据要求

- **格式**：CSV 文件，包含 `step`、`lr`、`loss` 三列，例如 `0,0.0003,2.0`。
- **位置**：由 `FOLDER_PATHS` 指定，例如 `./loss_curve_repo/csv_400/`。
- **命名**：必须匹配 `config.py` 中的 `TRAIN_SET` 和 `TEST_SET`，例如 `cosine_24000.csv`。

## 自定义

- **添加调度器**：扩展 `lr_schedules.py` 并更新 `data_loader.py`。
- **修改模型**：调整 `models.py` 以支持其他公式形式。
- **调节超参数**：通过 `main.py` 编辑或传入 `fitting.py` / `optimization.py` 中的参数。

## 贡献

1. Fork 本仓库。
2. 创建分支：`git checkout -b feature/your-feature`。
3. 提交修改：`git commit -m "Add your feature"`。
4. 推送分支：`git push origin feature/your-feature`。
5. 提交 pull request。

贡献时请同时包含测试和文档更新。

## 许可证

MIT License（见 `LICENSE` 文件）。

## 致谢

- 本项目为深度学习优化研究而开发。
- 基于 PyTorch、NumPy 和其他开源工具构建。
- 优化脚本归功于 [Kaifeng Lyu](https://github.com/vfleaking)。

## 联系方式

如有问题或遇到故障，请提交 GitHub issue，或发送邮件至 `luokr2002@outlook.com`。

使用 Multi-Power Law 调度来优化你的训练！

## Final Project 实验

本仓库还包含课程 final project 的可复现实验：**Predicting LLM Pretraining Loss Curves across Learning-Rate Schedules**。实验只用 cosine 学习率调度拟合 loss-curve predictor，然后评估到 WSD/WSDLD 等跨 schedule 曲线，重点观察学习率衰减阶段的外推误差。

Residual correction 分为两类：full ridge 和 FSL-light。full ridge 使用更丰富的 schedule 特征；如果它在 WSD/WSDLD 上失败，这应被解读为过拟合诊断结果，而不是代码 bug。FSL-light 是更轻量、可解释的替代方案，只使用 `log_tau`、`eta_norm`、`decay_conv`、`is_decay` 等少量 schedule functional 来修正 base predictor 的残差。

### Direction C / NCPL-style surrogate

Direction C 增加了 `scripts/run_ncpl_surrogate.py`。它是一个直接神经 surrogate，使用 model size、step progress 和 learning-rate schedule functionals 来直接预测 log loss。它会输出：

- `results/metrics/ncpl_surrogate_metrics.csv`
- `results/metrics/ncpl_surrogate_stage_metrics.csv`
- `results/predictions/ncpl_surrogate_cosine_train_<size>_ncpl_surrogate_<curve>.csv`
- `results/figures/ncpl_surrogate_comparison.png`

由于课程数据集里的 schedule 数量很少，这个方向应被解读为 high-risk baseline / future-work，而不是稳定结论。

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

主要指标输出：

```text
results/metrics/reproduction_metrics.csv
results/metrics/cross_schedule_metrics.csv
results/metrics/stage_metrics.csv
results/metrics/fsl_light_metrics.csv
results/metrics/alpha_sweep_metrics.csv
results/metrics/mpl_multistart_metrics.csv
results/metrics/mpl_multistart_summary.csv
results/metrics/stage_sensitivity_metrics.csv
results/metrics/ncpl_surrogate_metrics.csv
results/metrics/ncpl_surrogate_stage_metrics.csv
results/metrics/ablation_metrics.csv
results/metrics/correction_coefficients.csv
```

主要预测输出：

```text
results/predictions/ncpl_surrogate_cosine_train_<size>_ncpl_surrogate_<curve>.csv
```

主要图表输出：

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

更完整的中文说明见 `docs/final_project_overview_zh.md`。


