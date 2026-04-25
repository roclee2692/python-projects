# California Housing Linear Regression

English | [中文](#加州房价线性回归)

## English

This project implements a basic machine learning regression pipeline using the California Housing dataset.

### Goal

Predict median house value based on housing-related features.

### Workflow

1. Load dataset
2. Split train/test data
3. Standardize features
4. Train Linear Regression model
5. Evaluate with MAE, MSE, RMSE, and R2
6. Visualize prediction results and residuals

### Dataset

California Housing dataset from `sklearn.datasets.fetch_california_housing`.

### Run

```bash
pip install -r requirements.txt
python src/train_linear_regression.py
```

If you use the local Conda environment:

```bash
/mnt/c/Users/Raelon/miniconda3/envs/ai_env/python.exe src/train_linear_regression.py
```

### Metrics

- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R2: Coefficient of Determination

Current experiment results:

| Metric | Value |
| --- | ---: |
| MAE | 0.5332 |
| MSE | 0.5559 |
| RMSE | 0.7456 |
| R2 | 0.5758 |

### Outputs

The script saves figures to `outputs/figures/`:

- `true_vs_predicted.png`
- `residual_plot.png`

## 加州房价线性回归

本项目使用 California Housing 数据集实现一个基础的机器学习回归流程。

### 项目目标

根据房屋相关特征预测加州地区的房屋价值中位数。

### 实验流程

1. 加载数据集
2. 划分训练集和测试集
3. 标准化特征
4. 训练线性回归模型
5. 使用 MAE、MSE、RMSE 和 R2 评估模型
6. 可视化真实值与预测值，以及残差分布

### 数据集

数据集来自 `sklearn.datasets.fetch_california_housing`，无需手动下载，适合开源项目复现。

### 运行方法

```bash
pip install -r requirements.txt
python src/train_linear_regression.py
```

如果使用本地 Conda 环境：

```bash
/mnt/c/Users/Raelon/miniconda3/envs/ai_env/python.exe src/train_linear_regression.py
```

### 评估指标

- MAE：平均绝对误差
- MSE：均方误差
- RMSE：均方根误差
- R2：决定系数

当前实验结果：

| 指标 | 数值 |
| --- | ---: |
| MAE | 0.5332 |
| MSE | 0.5559 |
| RMSE | 0.7456 |
| R2 | 0.5758 |

### 输出文件

脚本会将图像保存到 `outputs/figures/`：

- `true_vs_predicted.png`
- `residual_plot.png`
