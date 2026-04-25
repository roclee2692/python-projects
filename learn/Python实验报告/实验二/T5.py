# 题目5：生成斐波那契数列
# 描述：编写函数，传入整数n，返回前n项斐波那契数列

def fibonacci(n):
    """
    生成斐波那契数列的前n项
    参数：n - 项数（整数）
    返回：包含前n项的列表
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    fib_list = [0, 1]
    for i in range(2, n):
        # 新项 = 前两项之和
        fib_list.append(fib_list[i-1] + fib_list[i-2])
    
    return fib_list


print("---- T5 Result ----")
print("斐波那契数列生成器\n")

# 获取用户输入
try:
    n = int(input("请输入要生成的项数（n >= 1）："))
    
    if n < 1:
        print("请输入大于等于1的整数！")
    else:
        result = fibonacci(n)
        print(f"\n前{n}项斐波那契数列：")
        print(result)
        
        # 显示具体的数列
        print(f"\n详细展示：")
        for i, num in enumerate(result, 1):
            print(f"第{i}项：{num}")

except ValueError:
    print("输入错误：请输入有效的整数")