"""
决策树与随机森林 - 完整实现
包含决策树、随机森林以及超参数调优
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.datasets import load_iris, load_breast_cancer
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 数据生成 ==========

def generate_classification_data(n_samples=200, random_state=42):
    """生成分类数据"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2) * 2
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) | ((X[:, 0] < -1) & (X[:, 1] < -1))
    return X.astype(float), y.astype(int)


# ========== 2. 评估函数 ==========

def evaluate_classifier(y_true, y_pred, y_proba=None, model_name="Model"):
    """评估分类模型"""
    print(f"\n{model_name} 评估结果:")
    print("=" * 50)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数:            {f1:.4f}")

    if y_proba is not None:
        auc_score = roc_auc_score(y_true, y_proba)
        print(f"AUC分数:           {auc_score:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_decision_boundary(X, y, model, title="决策边界"):
    """绘制决策边界"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', alpha=0.5, label='正常')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='s', alpha=0.5, label='异常')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ========== 3. 示例1：简单决策树 ==========

def example1_simple_decision_tree():
    """示例1：简单决策树"""
    print("="*50)
    print("示例1：简单决策树")
    print("="*50)

    # 生成数据
    X, y = generate_classification_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练决策树
    print("\n训练决策树...")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    # 预测
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)[:, 1]

    # 评估
    evaluate_classifier(y_test, y_pred, y_proba, "决策树 (depth=5)")

    # 可视化树
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=['特征1', '特征2'], class_names=['正常', '异常'],
              filled=True, rounded=True, fontsize=10)
    plt.title('决策树结构')
    plt.show()

    # 决策边界
    plot_decision_boundary(X_test, y_test, dt, "决策树决策边界")

    # 特征重要性
    print(f"\n特征重要性:")
    print(f"特征1: {dt.feature_importances_[0]:.4f}")
    print(f"特征2: {dt.feature_importances_[1]:.4f}")


# ========== 4. 示例2：深度对过拟合的影响 ==========

def example2_depth_effect():
    """示例2：树深度对过拟合的影响"""
    print("\n" + "="*50)
    print("示例2：树深度对过拟合的影响")
    print("="*50)

    # 生成数据
    X, y = generate_classification_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 测试不同深度
    depths = range(1, 16)
    train_scores = []
    test_scores = []

    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_scores.append(dt.score(X_train, y_train))
        test_scores.append(dt.score(X_test, y_test))

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, 'o-', label='训练准确率')
    plt.plot(depths, test_scores, 's-', label='测试准确率')
    plt.xlabel('树的深度')
    plt.ylabel('准确率')
    plt.title('树深度对过拟合的影响')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 找最优深度
    best_depth = depths[np.argmax(test_scores)]
    print(f"\n最优树深度: {best_depth}")
    print(f"该深度下的测试准确率: {test_scores[best_depth-1]:.4f}")


# ========== 5. 示例3：随机森林 ==========

def example3_random_forest():
    """示例3：随机森林"""
    print("\n" + "="*50)
    print("示例3：随机森林")
    print("="*50)

    # 生成数据
    X, y = generate_classification_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林
    print("\n训练随机森林...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 预测
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # 评估
    evaluate_classifier(y_test, y_pred, y_proba, "随机森林")

    # 决策边界
    plot_decision_boundary(X_test, y_test, rf, "随机森林决策边界")

    # 特征重要性
    print(f"\n特征重要性:")
    print(f"特征1: {rf.feature_importances_[0]:.4f}")
    print(f"特征2: {rf.feature_importances_[1]:.4f}")


# ========== 6. 示例4：决策树 vs 随机森林 ==========

def example4_comparison():
    """示例4：决策树 vs 随机森林对比"""
    print("\n" + "="*50)
    print("示例4：决策树 vs 随机森林对比")
    print("="*50)

    # 加载虹膜数据集
    data = load_iris()
    X = data.data
    y = data.target
    
    # 二分类（只取前两个类别）
    mask = y < 2
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 决策树
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_score = accuracy_score(y_test, dt_pred)

    # 随机森林
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_score = accuracy_score(y_test, rf_pred)

    print(f"\n决策树准确率: {dt_score:.4f}")
    print(f"随机森林准确率: {rf_score:.4f}")

    # 树的数量对性能的影响
    n_trees = range(1, 101, 5)
    scores = []

    for n in n_trees:
        rf_temp = RandomForestClassifier(n_estimators=n, max_depth=3, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train, y_train)
        scores.append(rf_temp.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, scores, 'o-')
    plt.xlabel('树的数量')
    plt.ylabel('准确率')
    plt.title('随机森林中树的数量对性能的影响')
    plt.grid(True)
    plt.show()


# ========== 7. 示例5：超参数调优 ==========

def example5_hyperparameter_tuning():
    """示例5：使用GridSearchCV调优超参数"""
    print("\n" + "="*50)
    print("示例5：超参数调优（GridSearchCV）")
    print("="*50)

    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print("\n开始网格搜索... (这可能需要几分钟)")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"\n最佳超参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 最优模型
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"测试准确率: {test_score:.4f}")


# ========== 8. 示例6：特征重要性分析 ==========

def example6_feature_importance():
    """示例6：特征重要性分析"""
    print("\n" + "="*50)
    print("示例6：特征重要性分析")
    print("="*50)

    # 加载虹膜数据集
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n特征重要性排序:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.title("特征重要性")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('重要性')
    plt.tight_layout()
    plt.show()


# ========== 主函数 ==========

if __name__ == "__main__":
    # 运行所有示例
    example1_simple_decision_tree()
    example2_depth_effect()
    example3_random_forest()
    example4_comparison()
    example5_hyperparameter_tuning()
    example6_feature_importance()

    print("\n" + "="*50)
    print("所有示例运行完成！")
    print("="*50)

