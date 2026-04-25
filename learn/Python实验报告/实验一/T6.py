# 题目6：进度条实现 - 非刷新式和动态刷新式
# 描述：实现两种进度条显示方式

import time

# ========== 方式1：非刷新文本进度条 ==========
# 特点：逐行打印，不刷新已有内容
print("---- Non-refresh Text Progress Bar ----")
for i in range(11):
    percent = i * 10
    bar = "#" * i + '-' * (10 - i)  # '#'表示完成，'-'表示未完成
    print(f"[{bar}] {percent}%")
    time.sleep(0.2)  # 模拟耗时操作

# ========== 方式2：单行动态刷新进度条 ==========
# 特点：在同一行动态更新，显示更新速度
print("\n---- Single-line Dynamic Refresh Progress Bar ----")

start_time = time.perf_counter()  # 记录开始时间
for i in range(101):
    percent = i
    bar = "#" * (i // 5) + '-' * (20 - i // 5)  # 构建进度条
    elapsed_time = time.perf_counter() - start_time  # 计算已用时间
    
    # 核心技术：\r 回到行首，end="" 禁止换行，flush=True 强制立即输出
    print(f"\r[{bar}] {percent:3d}% Time elapsed: {elapsed_time:.2f}s", end="", flush=True)
    time.sleep(0.05)  # 模拟耗时操作

print()  # 循环结束后换行 