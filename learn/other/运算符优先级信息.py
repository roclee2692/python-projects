import matplotlib.pyplot as plt

# 运算符优先级信息（部分常用）
levels = [
    ("()", "括号 (强制优先级)", 1),
    ("++ -- + - !", "一元运算", 2),
    ("* / %", "乘 除 模", 3),
    ("+ -", "加 减", 4),
    ("<< >>", "位移", 5),
    ("< <= > >=", "大小比较", 6),
    ("== !=", "相等/不等", 7),
    ("&", "按位与", 8)
    ("^", "按位异或", 9),
    ("|", "按位或", 10),
    ("&&", "逻辑与", 11),
    ("||", "逻辑或", 12),
    ("?:", "三目条件", 13),
    ("= += -= ...", "赋值", 14),
    (",", "逗号表达式", 15)
]
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')
table_data = [("优先级", "运算符", "含义")]
for op, meaning, level in levels:
    table_data.append((str(level), op, meaning))
# 绘制表格
table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# 设置标题
plt.title("C++ 常用运算符优先级表", fontsize=14, weight='bold')

plt.tight_layout()
plt.show()
