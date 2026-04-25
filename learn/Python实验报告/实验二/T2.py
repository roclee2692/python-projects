# 题目2：计算字符在每个单词中出现的次数
# 描述：给定句子和一个字符，统计该字符在每个单词中出现的次数

sentence = input("请输入一个句子：")
target_char = input("请输入要查找的字符：")

# 按空格分割单词（自动处理多个空格）
words = sentence.split()

print("---- T2 Result ----")
print(f"句子：{sentence}")
print(f"查找字符：'{target_char}'\n")

# 遍历每个单词，统计字符出现次数
for word in words:
    count = word.count(target_char)
    print(f"单词 '{word}' 中出现 {count} 次")