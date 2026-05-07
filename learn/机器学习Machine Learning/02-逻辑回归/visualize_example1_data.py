"""
可视化示例1的数据集
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据（与示例1相同）
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# 分离两个类别
X_class0 = X[y == 0]
X_class1 = X[y == 1]

print("="*60)
print("示例1：数据集信息")
print("="*60)

print(f"\n数据维度:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")

print(f"\n样本分布:")
print(f"  类别0（负类）: {len(X_class0)}个样本")
print(f"  类别1（正类）: {len(X_class1)}个样本")

print(f"\n特征统计:")
print(f"  Feature 0 (特征1):")
print(f"    最小值: {X[:, 0].min():.3f}")
print(f"    最大值: {X[:, 0].max():.3f}")
print(f"    平均值: {X[:, 0].mean():.3f}")
print(f"    标准差: {X[:, 0].std():.3f}")

print(f"\n  Feature 1 (特征2):")
print(f"    最小值: {X[:, 1].min():.3f}")
print(f"    最大值: {X[:, 1].max():.3f}")
print(f"    平均值: {X[:, 1].mean():.3f}")
print(f"    标准差: {X[:, 1].std():.3f}")

print(f"\n前10个样本:")
print(f"{'样本':^6} | {'Feature 0':^12} | {'Feature 1':^12} | {'标签':^4}")
print("-" * 50)
for i in range(10):
    label = "负类" if y[i] == 0 else "正类"
    print(f"{i:^6} | {X[i, 0]:^12.4f} | {X[i, 1]:^12.4f} | {label:^4}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：散点图
ax = axes[0]
ax.scatter(X_class0[:, 0], X_class0[:, 1], c='red', label='类别0（负类）', alpha=0.6, s=50)
ax.scatter(X_class1[:, 0], X_class1[:, 1], c='blue', label='类别1（正类）', alpha=0.6, s=50)
ax.set_xlabel('Feature 0（特征1）', fontsize=12)
ax.set_ylabel('Feature 1（特征2）', fontsize=12)
ax.set_title('示例1：数据分布（散点图）', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 图2：直方图分布
ax = axes[1]
ax.hist(X[y == 0, 0], alpha=0.5, label='类别0 - Feature 0', bins=20, color='red')
ax.hist(X[y == 1, 0], alpha=0.5, label='类别1 - Feature 0', bins=20, color='blue')
ax.set_xlabel('Feature 0 的值', fontsize=12)
ax.set_ylabel('频率', fontsize=12)
ax.set_title('特征0的分布', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n✅ 可视化完成！")
