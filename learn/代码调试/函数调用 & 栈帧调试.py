def multiply(a, b):
    """乘法函数"""
    result = a * b
    return result


def add(a, b):
    """加法函数"""
    result = a + b
    return result


def calculate_expression(x, y):
    """计算表达式 (x + y) * (x - y)"""
    sum_part = add(x, y)       # 断点可以打在这里
    diff_part = add(x, -y)
    final_result = multiply(sum_part, diff_part)
    return final_result


# 测试用例
if __name__ == "__main__":
    num1 = 5
    num2 = 3
    result = calculate_expression(num1, num2)
    print(f"({num1} + {num2}) * ({num1} - {num2}) = {result}")