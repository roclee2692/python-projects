# 题目6：统计随机数中出现5的次数
# 描述：随机生成100次1~10的整数，统计出现5的次数

import random

def count_five(times=100):
    """
    随机生成指定次数的1~10整数，统计出现5的次数
    参数：times - 生成的随机数次数（默认100次）
    返回：(随机数列表, 出现5的次数)
    """
    # 生成100个1~10的随机整数
    random_numbers = [random.randint(1, 10) for _ in range(times)]
    
    # 统计出现5的次数
    count = random_numbers.count(5)
    
    return random_numbers, count


# 主程序
print("---- T6 Result ----")
print("随机数中出现5的次数统计\n")

# 生成随机数并统计
numbers, count = count_five(100)

print(f"生成的100个随机数：")
print(numbers)

print(f"\n统计结果：")
print(f"出现5的次数：{count}")
print(f"出现的频率：{count}% ")

# 显示各数字的出现次数
print(f"\n各数字出现的次数分布：")
for i in range(1, 11):
    freq = numbers.count(i)
    print(f"数字{i}：出现{freq}次")

