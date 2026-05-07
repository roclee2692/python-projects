"""
鸢尾花（Iris）数据集 - 多分类示例
===========================================
目标：分类 3 个类别（Setosa, Versicolor, Virginica）
使用：逻辑回归的 one-vs-rest (OvR) 策略

多分类原理：
  - 为每个类别训练一个二分类器（该类 vs 其他类）
  - 对新样本的预测，选择概率最高的类别
  - P(y|x) 通过 softmax 或直接概率得到
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载与准备 ====================
print("=" * 60)
print("鸢尾花多分类：3个类别的分类")
print("=" * 60)

# 加载数据
iris = load_iris()
X = iris.data  # 特征：(150, 4)
y = iris.target  # 标签：0=Setosa, 1=Versicolor, 2=Virginica
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

print(f"\n原始数据shape: {X.shape}")
print(f"类别分布:")
for i, name in enumerate(target_names):
    print(f"  - {name} ({i}): {np.sum(y == i)}")

# ==================== 2. 特征选择与可视化 ====================
# 使用2个特征便于可视化：花萼长度(Sepal Length) 和 花瓣长度(Petal Length)
feature_indices = [0, 2]  # Sepal Length, Petal Length
X_2d = X[:, feature_indices]

print(f"\n选择特征：{iris.feature_names[feature_indices[0]]} 和 {iris.feature_names[feature_indices[1]]}")

# ==================== 3. 数据标准化 ====================
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

# ==================== 4. 数据划分 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_2d_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ==================== 5. 训练多分类逻辑回归模型 ====================
print("\n" + "=" * 60)
print("训练多分类逻辑回归模型 (one-vs-rest)")
print("=" * 60)

# multi_class='ovr' 表示 one-vs-rest 策略
model = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print(f"\n模型参数:")
print(f"  - 共有 {len(model.coef_)} 个二分类器")
print(f"  - 每个分类器的权重:")
for i, name in enumerate(target_names):
    print(f"    {name}: w={model.coef_[i]}, b={model.intercept_[i]:.4f}")

# ==================== 6. 预测与评估 ====================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n性能评估:")
print(f"  - 训练集准确率: {train_acc:.4f}")
print(f"  - 测试集准确率: {test_acc:.4f}")

# 获取预测概率
y_pred_proba = model.predict_proba(X_test)
print(f"\n预测概率示例 (前5个样本):")
for i in range(min(5, len(y_test))):
    probs = y_pred_proba[i]
    print(f"  样本{i}: ", end="")
    for j, name in enumerate(target_names):
        print(f"P({name})={probs[j]:.4f} ", end="")
    print(f"| 真标签={target_names[y_test[i]]}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n混淆矩阵:")
print(cm)

# 分类报告
print(f"\n分类报告:")
print(classification_report(y_test, y_pred_test, target_names=target_names))

# ==================== 7. 决策边界可视化 ====================
print("\n生成决策边界图...")

# 创建网格
h = 0.02
x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 预测网格上所有点的标签
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 定义颜色
colors = ['lightblue', 'lightgreen', 'lightcoral']

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：决策边界
ax = axes[0]
for i in range(len(target_names)):
    ax.contourf(xx, yy, (Z == i).astype(int), levels=[0.5, 1.5], colors=[colors[i]], alpha=0.4)

# 绘制训练点
for i, name in enumerate(target_names):
    ax.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
              c=colors[i], marker='o', label=f'{name} (Train)', s=50, edgecolors='k', alpha=0.7)

# 绘制测试点
for i, name in enumerate(target_names):
    ax.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
              c=colors[i], marker='s', label=f'{name} (Test)', s=100, edgecolors='black')

ax.set_xlabel(iris.feature_names[feature_indices[0]], fontsize=11)
ax.set_ylabel(iris.feature_names[feature_indices[1]], fontsize=11)
ax.set_title('Logistic Regression Decision Boundary (Multi-class)\n逻辑回归决策边界 (多分类)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# 右图：混淆矩阵热力图
ax = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
           xticklabels=target_names,
           yticklabels=target_names)
ax.set_ylabel('True Label / 真标签')
ax.set_xlabel('Predicted Label / 预测标签')
ax.set_title(f'Confusion Matrix / 混淆矩阵 (Accuracy: {test_acc:.2%})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('iris_multiclass_classification.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：iris_multiclass_classification.png")
plt.show()

# ==================== 8. 每类预测概率分布 ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, name in enumerate(target_names):
    ax = axes[i]
    proba_class = y_pred_proba[:, i]
    
    ax.hist(proba_class[y_test == i], bins=15, alpha=0.6, label=f'True {name}', color='blue', edgecolor='black')
    ax.hist(proba_class[y_test != i], bins=15, alpha=0.6, label=f'Other classes', color='red', edgecolor='black')
    
    ax.set_xlabel(f'P({name}|x)', fontsize=11)
    ax.set_ylabel('Frequency / 频数')
    ax.set_title(f'Class "{name}" Probability Distribution\n类别"{name}"的预测概率分布', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('iris_probability_distribution.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：iris_probability_distribution.png")
plt.show()

# ==================== 9. 2D概率热力图 ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, name in enumerate(target_names):
    Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, i]
    Z_proba = Z_proba.reshape(xx.shape)
    
    ax = axes[i]
    contourf = ax.contourf(xx, yy, Z_proba, levels=15, cmap='RdYlBu_r', alpha=0.8)
    
    # 绘制该类的样本
    ax.scatter(X_2d_scaled[y == i, 0], X_2d_scaled[y == i, 1],
              c='black', marker='o', s=30, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel(iris.feature_names[feature_indices[0]], fontsize=10)
    ax.set_ylabel(iris.feature_names[feature_indices[1]], fontsize=10)
    ax.set_title(f'P({name}|x) Probability Heatmap\nP({name}|x)的热力图', fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('概率', fontsize=9)

plt.tight_layout()
plt.savefig('iris_probability_heatmap_multiclass.png', dpi=150, bbox_inches='tight')
print("✓ 已保存：iris_probability_heatmap_multiclass.png")
plt.show()

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
