import numpy as np
# a=np.array([[2,4,6],[2,4,5]])
# b=np.array([1,4,6,7,8,8,7,4])
# print(a)
# print(b)
# z=np.zeros((3,4))
# o=np.ones([2,2])
# print(z)
# print(o)
# # 单位矩阵
# I = np.eye(8)
# print(I)
# r1=np.arange(-2,10,2)
# print(r1)
# r2=np.linspace(-1,11,10)
# print(r2)
# x=np.random.rand(4,6)
# print(x)
# 打印数组属性
# print("x.ndim =", x.ndim)
# print("x.shape =", x.shape)
# print("x.size =", x.size)
# print("x.dtype =", x.dtype)
# print("x.itemsize =", x.itemsize)# 每个元素字节数，8
# print("x.nbytes =", x.nbytes)#数组 x 所有元素占用的总字节数，等于 x.size * x.itemsize。
# print("x.T =\n", x.T)#返回 x 的转置（对二维以上数组按维度逆序）。 行列交换
# print("x.flatten() =", x.flatten())#将 x 展平成一维数组，返回新拷贝，不改变原数组。

# 正确的 reshape
# print("x.reshape(2, 3, 4) =\n", x.reshape(2, 3, 4))
# # 或者自动推断最后一维
# print("x.reshape(2, 3, -1) =\n", x.reshape(2, 3, -1))
# b = np.arange(10)       # [0,1,2,…,9]
# b[2:7:2]                # [2,4,6]
# M = np.arange(12).reshape(3,4)
# M[:,1:3]                # 所有行，第 1 到 2 列
# d = np.arange(9).reshape(3,3)
# print(d)
# idx = [2,0]
# d[idx]                  # 选第 2 行与第 0 行，结果形状 (2,3)
# d[:,idx]              # 每行取第 2 列与第 0 列
# print(d[idx])
# print(d[:,[2,0]])

# a=np.array([1,2,3,4,5,6,7,8,9])
# b=np.array([3,4,5,6,2,1,4,6,7])
# print(a+b)
# print(a*b)
# print(a**2)
# print(np.sin(a))
# print(np.cos(b))
# print(np.exp(a))
# arr=np.exp(a)
# for x in arr:
#     print(f"{x:.8f}")
# np.set_printoptions(suppress=True)  # 禁用科学计数法
# print(np.exp(b))
# print(np.log(a))
# A=np.array([[1,2,3],[4,5,6],[7,8,9]])
# v=np.array([1,10,100])
# print(A+v)
# A = np.array([[1,2,3],
# #               [4,5,6]])      # shape (2,3)
# # B = np.array([[7, 8],
# #               [9,10],
# #               [11,12]])      # shape (3,2)
# # C1 = A.dot(B)               # np.dot(A,B)，shape (2,2)
# # C2 = A @ B                  # 等价于 dot
# # print(C1)
# # print(C2)
# # print(B.dot(A))
# # print(B@A)
M = np.array([[3,1],
              [1,2]])
b = np.array([9,8])
x = np.linalg.solve(M, b)   # 求 x，使 M·x = b
print(x)