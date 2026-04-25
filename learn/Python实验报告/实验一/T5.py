# 题目5：三角函数应用 - 斜边长度计算
# 描述：已知高度和角度，计算斜边长度 (正弦函数应用)
# 公式：lenght = height / sin(angle)

from math import radians, sin

height = float(input("please input a height: "))
angle = float(input("please input a angle: "))

# 将角度从度转换为弧度
rad = radians(angle)

# 根据三角函数关系计算斜边长度
length = height / sin(rad)

print("---- T5 Result ----")
print(f"length is: {length:.2f}")