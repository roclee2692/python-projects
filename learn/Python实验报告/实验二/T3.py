# 题目3：求一元二次方程的根
# 描述：输入系数a、b、c，求方程ax^2+bx+c=0的根
# 公式：x = (-b ± √(b²-4ac)) / 2a

import math

def solve(a, b, c):
    """
    求一元二次方程的根
    参数：a, b, c - 方程的系数
    返回：根的信息（元组或字符串）
    """
    # 判断a是否为0（不是二次方程）
    if a == 0:
        if b == 0:
            return "不是方程" if c != 0 else "无穷多解"
        else:
            # 一次方程 bx + c = 0
            x = -c / b
            return f"一次方程的根：x = {x:.2f}"
    
    # 计算判别式 Δ = b² - 4ac
    delta = b * b - 4 * a * c
    
    # 根据判别式分情况讨论
    if delta > 0:
        # 两个不同的实根
        x1 = (-b + math.sqrt(delta)) / (2 * a)
        x2 = (-b - math.sqrt(delta)) / (2 * a)
        return f"两个不同的实根：x1 = {x1:.2f}，x2 = {x2:.2f}"
    
    elif delta == 0:
        # 一个重根
        x = -b / (2 * a)
        return f"一个重根：x = {x:.2f}"
    
    else:
        # 无实根（复数根）
        real_part = -b / (2 * a)
        imag_part = math.sqrt(-delta) / (2 * a)
        return f"无实根（复数根）：x1 = {real_part:.2f} + {imag_part:.2f}i，x2 = {real_part:.2f} - {imag_part:.2f}i"


# 主程序
print("---- T3 Result ----")
print("求一元二次方程 ax²+bx+c=0 的根\n")

print("---- 手动输入测试 ----")
try:
    a = float(input("请输入系数a："))
    b = float(input("请输入系数b："))
    c = float(input("请输入系数c："))
    
    result = solve(a, b, c)
    print(f"\n方程：{a}x²+{b}x+{c}=0")
    print(f"结果：{result}")
except ValueError:
    print("输入错误：请输入有效的数字")
