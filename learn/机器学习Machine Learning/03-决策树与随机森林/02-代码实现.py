"""
决策树与随机森林 - 完整实现
包含决策树、随机森林以及超参数调优
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 避免 tkinter 警告
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 数据生成 ==========

def generate_classification_data(n_samples=None, random_state=42):
    """使用真实房价数据集 (California Housing)"""
    np.random.seed(random_state)

    # 加载California Housing数据集（保留特征名称）
    data = fetch_california_housing(as_frame=True)
    X = data.data.values
    y = data.target.values

    # 将房价转为二分类：高于中位数为1（高价），否则为0（低价）
    y = (y > np.median(y)).astype(int)

    # 如果指定了n_samples，则随机采样
    if n_samples is not None and n_samples < len(X):
        indices = np.random.RandomState(random_state).choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

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
    print("示例1：简单决策树 (房价数据集)")
    print("="*50)

    # 加载真实数据
    X, y = generate_classification_data()
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

    # 可视化树结构（只用前两个最重要的特征，便于显示）
    feature_names = ['收入中位数', '房屋年龄', '平均房间数', '平均卧室数', '人口', '平均占用率', '纬度', '经度']
    top_features_idx = np.argsort(dt.feature_importances_)[::-1][:2]
    X_train_top = X_train[:, top_features_idx]
    X_test_top = X_test[:, top_features_idx]

    dt_simple = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_simple.fit(X_train_top, y_train)

    plt.figure(figsize=(25, 15))
    plot_tree(dt_simple,
              feature_names=[feature_names[i] for i in top_features_idx],
              class_names=['低价', '高价'],
              filled=True, rounded=True, fontsize=10)
    plt.title('决策树结构（使用前两个重要特征）')
    plt.tight_layout()
    plt.show()

    # 完整特征重要性
    print(f"\n特征重要性:")
    for i in range(len(feature_names)):
        print(f"{feature_names[i]}: {dt.feature_importances_[i]:.4f}")


# ========== 4. 示例2：深度对过拟合的影响 ==========

def example2_depth_effect():
    """示例2：树深度对过拟合的影响"""
    print("\n" + "="*50)
    print("示例2：树深度对过拟合的影响 (房价数据集)")
    print("="*50)

    # 加载真实数据
    X, y = generate_classification_data()
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
    print("示例3：随机森林 (房价数据集)")
    print("="*50)

    # 加载真实数据
    X, y = generate_classification_data()
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

    # 特征重要性
    print(f"\n特征重要性:")
    feature_names = ['收入中位数', '房屋年龄', '平均房间数', '平均卧室数', '人口', '平均占用率', '纬度', '经度']
    for i in range(len(feature_names)):
        print(f"{feature_names[i]}: {rf.feature_importances_[i]:.4f}")


# ========== 6. 示例4：决策树 vs 随机森林 ==========

def example4_comparison():
    """示例4：决策树 vs 随机森林对比"""
    print("\n" + "="*50)
    print("示例4：决策树 vs 随机森林对比 (Iris 3分类 - 只用前两个特征)")
    print("="*50)

    # 加载Iris数据集（3分类）
    iris_data = pd.read_csv(r"D:\DpanPython\python-projects\learn\机器学习Machine Learning\数据集\iris.csv")
    # 只用前两个特征，增加难度
    X = iris_data[['SepalLengthCm', 'SepalWidthCm']].values
    y = iris_data['Species'].values

    # 编码标签
    species_map = {'Iris-setosa': 0, 'Iris-versicolour': 1, 'Iris-virginica': 2}
    y = np.array([species_map[s] for s in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 决策树
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_score = accuracy_score(y_test, dt_pred)

    # 随机森林
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_score = accuracy_score(y_test, rf_pred)

    print(f"\n决策树准确率: {dt_score:.4f}")
    print(f"随机森林准确率: {rf_score:.4f}")

    print(f"\n决策树分类报告:")
    print(classification_report(y_test, dt_pred, target_names=['Setosa', 'Versicolour', 'Virginica']))

    print(f"\n随机森林分类报告:")
    print(classification_report(y_test, rf_pred, target_names=['Setosa', 'Versicolour', 'Virginica']))

    # 树的数量对性能的影响（用交叉验证，更稳定）
    n_trees_list = [1, 2, 5, 10, 20, 50, 100, 150, 200]
    rf_cv_scores = []
    rf_test_scores = []

    print(f"\n评估不同树数的性能:")
    for n in n_trees_list:
        rf_temp = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42, n_jobs=-1)
        cv_score = cross_val_score(rf_temp, X_train, y_train, cv=5).mean()
        rf_temp.fit(X_train, y_train)
        test_score = rf_temp.score(X_test, y_test)
        rf_cv_scores.append(cv_score)
        rf_test_scores.append(test_score)
        print(f"  树数={n:3d}: 交叉验证={cv_score:.4f}, 测试集={test_score:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(n_trees_list, rf_cv_scores, 'o-', label='交叉验证准确率', linewidth=2)
    plt.plot(n_trees_list, rf_test_scores, 's-', label='测试集准确率', linewidth=2)
    plt.xlabel('森林中的树数量')
    plt.ylabel('准确率')
    plt.title('随机森林中树的数量对性能的影响')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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

