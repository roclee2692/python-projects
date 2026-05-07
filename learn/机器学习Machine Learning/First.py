import numpy as np

# 1. 构造数据集
# C1: y=1 只有1个样本 (1,1)
# C2: y=0 共12个：(1,0),(0,1),(0,0) 各4个
X = np.array([
    [1, 1],
    # 4个 (1,0)
    [1,0],[1,0],[1,0],[1,0],
    # 4个 (0,1)
    [0,1],[0,1],[0,1],[0,1],
    # 4个 (0,0)
    [0,0],[0,0],[0,0],[0,0]
])
y = np.array([1,
              0,0,0,0,
              0,0,0,0,
              0,0,0,0])

# 2. 定义sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. 初始化参数
w = np.zeros(2)
b = 0.0
lr = 0.5
epochs = 5000

# 4. 梯度下降迭代训练
n = len(X)
for i in range(epochs):
    # 前向传播
    z = np.dot(X, w) + b
    h = sigmoid(z)
    
    # 交叉熵损失
    loss = -np.mean(y * np.log(h+1e-8) + (1-y)*np.log(1-h+1e-8))
    
    # 梯度
    dw = (1/n) * np.dot(X.T, h - y)
    db = (1/n) * np.sum(h - y)
    
    # 更新
    w -= lr * dw
    b -= lr * db
    
    # 每500轮打印一次
    if i % 500 == 0:
        print(f"迭代轮数:{i:4d}  损失:{loss:.4f}  w:{w.round(3)}  b:{b:.3f}")

# 5. 测试样本 x_test = [1,1]
x_test = np.array([1,1])
z_test = np.dot(x_test, w) + b
p_test = sigmoid(z_test)
print("\n==== 最终训练结果 ====")
print(f"学到的权重 w = {w.round(3)}")
print(f"学到的偏置 b = {b:.3f}")
print(f"测试点 [1,1] 的 z 值 = {z_test:.3f}")
print(f"测试点属于C1的概率 = {p_test:.4f}")
print("分类结果：", "C1(类别1)" if p_test>0.5 else "C2(类别2)")