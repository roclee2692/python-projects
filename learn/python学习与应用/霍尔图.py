import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------------------- 1. 整理完整数据 ----------------------
raw_voltage = [
    0, 10, 21, 32, 45, 54, 68, 78, 91, 105, 116, 131, 144, 157, 176,
    188, 207, 229, 240, 263, 279, 304, 324, 343, 376, 394, 428, 453,
    478, 522, 546, 593, 628, 666, 724, 762, 824, 874, 926, 1003, 1056,
    1140, 1211, 1282, 1385, 1458, 1563, 1653, 1735, 1827, 1886, 1948, 1948
]
x_all = np.array([0.2 * i for i in range(len(raw_voltage))])  # 所有位移数据
U_all = np.array([val / 1000 for val in raw_voltage])        # 所有电压数据

# ---------------------- 2. 提取x=0-6mm区间数据（前31组） ----------------------
n_linear = 31  # 前31组对应x=0-6.0mm（0.2*(31-1)=6.0）
x_linear = x_all[:n_linear]
U_linear = U_all[:n_linear]

# ---------------------- 3. 对0-6mm区间做线性拟合 ----------------------
slope, intercept, r_value, p_value, std_err = stats.linregress(x_linear, U_linear)
U_fit = slope * x_linear + intercept  # 拟合线的电压值

# ---------------------- 4. 绘制图像（重点展示0-6mm拟合） ----------------------
plt.figure(figsize=(10, 6), dpi=100)
# 绘制所有实验数据散点
plt.scatter(x_all, U_all, color='deepskyblue', s=30, label='所有实验数据')
# 绘制0-6mm区间的线性拟合线（仅覆盖该区间）
plt.plot(x_linear, U_fit, color='crimson', linestyle='-', linewidth=2,
         label=f'x=0-6mm线性拟合 (R²={r_value**2:.4f})')
# 标注线性区间边界
plt.axvline(x=6.0, color='gray', linestyle='--', alpha=0.7, label='线性区间边界(x=6mm)')

# 图像标注
plt.xlabel('位移 x (mm)', fontsize=12)
plt.ylabel('电压 U (V)', fontsize=12)
plt.title('霍尔传感器x=0-6mm区间线性拟合图', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.xlim(-0.2, 10.5)  # 放大x轴范围，清晰展示区间
plt.show()

# 输出拟合结果
print(f"x=0-6mm区间拟合结果：")
print(f"灵敏度k（斜率）：{slope:.4f} V/mm")
print(f"拟合截距：{intercept:.4f} V")
print(f"线性相关系数R²：{r_value**2:.4f}")