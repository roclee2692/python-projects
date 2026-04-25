# 题目1：整数反转
# 描述：给定一个整数（可以为负），将各位数字反转得到新数，新数不得有前导0

def reverse_integer(num):
    """
    反转整数的各位数字
    参数：num - 待反转的整数
    返回：反转后的新整数
    """
    # 判断原数是否为负
    is_negative = num < 0
    
    # 转换为正数进行处理
    num = abs(num)
    
    # 将数字转换为字符串，反转后再转回整数
    reversed_num = int(str(num)[::-1])
    
    # 如果原数为负，返回负数
    if is_negative:
        reversed_num = -reversed_num
    
    return reversed_num

print("---- T1 Result ----")
# 交互式输入
print("\n---- 手动输入测试 ----")
user_input = int(input("请输入一个整数（可为负）："))
result = reverse_integer(user_input)
print(f"反转结果：{result}")
