# 题目4：复数的属性与运算
# 描述：计算复数的实部、虚部和绝对值（模长）

z = -3 + 4j              # 定义复数
real_part = z.real       # 提取实部
virtual_part = z.imag    # 提取虚部
abs_value = abs(z)       # 计算绝对值（模长）

print("---- T4 Result ----")
print(f"复数z = {z}")
print(f"实部：{real_part}")
print(f"虚部：{virtual_part}")
print(f"绝对值（模长）：{abs_value}")