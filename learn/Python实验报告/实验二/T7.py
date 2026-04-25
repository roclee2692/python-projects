# 题目7：装饰器实现函数运行时间统计
# 描述：编写装饰器，在函数结束时输出"Game Over"和运行时间

import time
from functools import wraps

def timer_decorator(func):
    """
    装饰器：统计函数运行时间并输出结束信息
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.time()
        
        # 执行原函数
        result = func(*args, **kwargs)
        
        # 记录结束时间
        end_time = time.time()
        
        # 计算运行时间
        run_time = end_time - start_time
        
        # 输出结果信息
        print(f"\n{'='*40}")
        print(f"函数名：{func.__name__}")
        print(f"运行时间：{run_time:.4f}秒")
        print("Game Over")
        print(f"{'='*40}\n")
        
        return result
    
    return wrapper


# 使用装饰器定义几个测试函数
@timer_decorator
def count_numbers(n):
    """计算1到n的和"""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total


@timer_decorator
def sleep_task(seconds):
    """等待指定秒数"""
    print(f"开始等待 {seconds} 秒...")
    time.sleep(seconds)
    print(f"等待完毕")


@timer_decorator
def fibonacci_sum(n):
    """计算斐波那契数列前n项的和"""
    if n <= 0:
        return 0
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return sum(fib)


# 主程序
print("---- T7 Result ----")
print("装饰器：函数运行时间统计\n")


# 交互测试
print("\n---- 交互测试 ----")
choice = input("要自定义测试吗? (y/n): ")
if choice.lower() == 'y':
    try:
        n = int(input("请输入整数n (用于计算1到n的和): "))
        print(f"\n计算1到{n}的和")
        result = count_numbers(n)
        print(f"结果：{result}")
    except ValueError:
        print("输入错误：请输入有效的整数")
