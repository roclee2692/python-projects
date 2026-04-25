# 题目1：计算坚持与懈怠的力量差异
# 描述：计算一个人每天进步1%和每天退步1%，经过365天后的差异

effort = pow(1.01, 365)      # 每天进步1%，365天后的结果
slack = pow(0.99, 365)        # 每天退步1%，365天后的结果
dif = effort - slack           # 计算两者之间的差异

print("---- T1 Result ----")
print(f"effort 1% everyday: {effort:.2f}")
print(f"slack 1% everyday: {slack:.2f}")
print(f"different is: {dif:.2f}")

# 注：pow()函数对比
# 方式          | 是否需要导入 | 支持整数/复数 | 支持三参数 mod | 返回值类型
# pow()         | 否          | 是           | 是            | 与输入一致
# **            | 否          | 是           | 否            | 与输入一致
# math.pow()    | 是          | 否(强制浮点)  | 否            | 总是浮点数