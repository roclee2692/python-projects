# 题目2：考虑休息日的年度能力变化
# 描述：在一年365天中，周末（周六、周日）退步1%，工作日进步1%

ability = 100              # 初始能力值
work_factor = 1.01         # 工作日能力增长因子（进步1%）
rest_factor = 0.99         # 休息日能力衰减因子（退步1%）

# 遍历365天，根据日期是否为休息日来更新能力值
for day in range(365):
    if day % 7 in (6, 0):  # 判断是否为周末（第6天和第0天）
        ability *= rest_factor  # 休息日能力衰减
    else:
        ability *= work_factor  # 工作日能力增长

print("---- T2 Result ----")
print(f"a year after, ability is: {ability:.2f}")