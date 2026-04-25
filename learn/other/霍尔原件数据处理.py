import numpy as np
from scipy import stats

# ⚠️ 注意：这里用的是 U(V) 列的数据（已经除以1000）
# 线性区间数据 (0~6mm, 序号1-31)
x_data = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
                   2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
                   4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])

# ✅ 正确数据：U(V) = 原始值 ÷ 1000
y_data = np.array([0.000, 0.010, 0.021, 0.032, 0.045, 0.054, 0.068, 0.078, 0.091, 0.105,
                   0.116, 0.131, 0.144, 0.157, 0.176, 0.188, 0.207, 0.229, 0.240, 0.263,
                   0.279, 0.304, 0.324, 0.343, 0.376, 0.394, 0.428, 0.453, 0.478, 0.522, 0.546])

print("数据验证：")
print(f"第1个点: x={x_data[0]}, U={y_data[0]} (原始值=0)")
print(f"第31个点: x={x_data[30]}, U={y_data[30]} (原始值=546)")
print()

# 基本统计量
n = len(x_data)
sum_x = np.sum(x_data)
sum_y = np.sum(y_data)
sum_x2 = np.sum(x_data**2)
sum_y2 = np.sum(y_data**2)
sum_xy = np.sum(x_data * y_data)
mean_x = np.mean(x_data)
mean_y = np.mean(y_data)

# 线性回归 - 手动计算
k_manual = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b_manual = mean_y - k_manual * mean_x

# 相关系数 - 手动计算
r_manual = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

# 线性回归 - 使用scipy验证
slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
R2 = r_value**2

# 打印结果
print("=" * 70)
print("线性区间数据统计 (x = 0.0 ~ 6.0 mm)")
print("=" * 70)
print(f"数据点数 n = {n}")
print(f"Σx = {sum_x:.4f} mm")
print(f"Σy = {sum_y:.6f} V")
print(f"Σx² = {sum_x2:.4f} mm²")
print(f"Σy² = {sum_y2:.8f} V²")
print(f"Σxy = {sum_xy:.6f} mm·V")
print(f"x̄ = {mean_x:.4f} mm")
print(f"ȳ = {mean_y:.6f} V")
print()
print("=" * 70)
print("线性回归参数（手动计算）")
print("=" * 70)
print(f"斜率 k = {k_manual:.6f} V/mm = {k_manual*1000:.4f} mV/mm")
print(f"截距 b = {b_manual:.6f} V")
print(f"相关系数 r = {r_manual:.8f}")
print(f"决定系数 R² = {r_manual**2:.8f}")
print()
print("=" * 70)
print("线性回归参数（scipy.stats.linregress验证）")
print("=" * 70)
print(f"斜率 k = {slope:.6f} V/mm = {slope*1000:.4f} mV/mm")
print(f"截距 b = {intercept:.6f} V")
print(f"相关系数 r = {r_value:.8f}")
print(f"决定系数 R² = {R2:.8f}")
print(f"标准误差 std_err = {std_err:.8f}")
print()
print("=" * 70)
print("拟合方程")
print("=" * 70)
print(f"U = {slope:.6f}x + {intercept:.6f}")
print(f"或写成: U = {slope:.4f}x + {intercept:.4f}")
print()

# 计算残差统计
y_fitted = slope * x_data + intercept
residuals = y_data - y_fitted
max_positive_residual = np.max(residuals)
max_negative_residual = np.min(residuals)
residual_std = np.std(residuals, ddof=1)
max_abs_residual = max(abs(max_positive_residual), abs(max_negative_residual))

print("=" * 70)
print("残差分析")
print("=" * 70)
print(f"最大正残差 = {max_positive_residual:.6f} V (位置: x={x_data[np.argmax(residuals)]:.1f}mm)")
print(f"最大负残差 = {max_negative_residual:.6f} V (位置: x={x_data[np.argmin(residuals)]:.1f}mm)")
print(f"最大绝对残差 = {max_abs_residual:.6f} V")
print(f"残差标准差 σ = {residual_std:.6f} V")
print(f"非线性度 = {max_abs_residual/y_data[-1]*100:.4f}%")
print()

# 显示部分数据对比表格
print("=" * 70)
print("部分数据对比表 (实测 vs 拟合)")
print("=" * 70)
print("序号\tx(mm)\tU实测(V)\tU拟合(V)\t残差(V)")
print("-" * 70)
for i in [0, 5, 10, 15, 20, 25, 30]:
    print(f"{i+1}\t{x_data[i]:.1f}\t{y_data[i]:.3f}\t\t{y_fitted[i]:.6f}\t{residuals[i]:+.6f}")

print("\n" + "=" * 70)
print("📋 填表用的最终结果汇总")
print("=" * 70)
print(f"【数据点数】 n = {n}")
print(f"【位移之和】 Σx = {sum_x:.1f} mm")
print(f"【电压之和】 Σy = {sum_y:.3f} V")
print(f"【位移平方和】 Σx² = {sum_x2:.2f} mm²")
print(f"【电压平方和】 Σy² = {sum_y2:.6f} V²")
print(f"【位移电压乘积和】 Σxy = {sum_xy:.4f} mm·V")
print(f"【平均位移】 x̄ = {mean_x:.1f} mm")
print(f"【平均电压】 ȳ = {mean_y:.3f} V")
print()
print(f"【斜率/灵敏度】 k = {slope:.4f} V/mm = {slope*1000:.2f} mV/mm")
print(f"【截距】 b = {intercept:.4f} V")
print(f"【相关系数】 r = {r_value:.6f}")
print(f"【决定系数】 R² = {R2:.6f}")
print()
print(f"【拟合方程】 U = {slope:.4f}x + {intercept:.4f}")
print("=" * 70)