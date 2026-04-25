# 题目4：计算1,2,3,4能组成多少个三位数
# 要求：互不相同、不重复

from itertools import permutations

# 给定的数字
digits = [1, 2, 3, 4]

# 使用permutations生成所有3位排列
three_digit_numbers = list(permutations(digits, 3))

# 转换为实际的三位数
numbers = [int(str(a) + str(b) + str(c)) for a, b, c in three_digit_numbers]

print("---- T4 Result ----")
print(f"能组成的三位数个数：{len(numbers)}")
print(f"\n所有互不相同的三位数（共{len(numbers)}个）：")
print(sorted(numbers))

confirm = input("\n是否显示验证？(y/n): ")
if confirm.lower() == 'y':
    print("\n验证信息：")
    print(f"从4个数字中选3个进行排列：P(4,3) = 4×3×2 = {4*3*2}")
    print(f"实际生成的个数：{len(numbers)}")
    print("✓ 验证成功" if len(numbers) == 24 else "✗ 验证失败")
