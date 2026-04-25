import os

# ✅ 你的项目根目录（路径已改好）
project_dir = "D:/DProject"

# ✅ 支持的文件扩展名（你常见的都包含了）
file_extensions = (
    ".py", ".cpp", ".c", ".h", ".txt", ".md", ".json", ".js",
    ".html", ".css", ".csv", ".xml", ".ini", ".log", ".bat"
)

# 用于记录转换状态
converted_files = []

skipped_files = []

for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.lower().endswith(+file_extensions):
            file_path = os.path.join(root, file)
            try:
                # 以 GBK 编码读取原文件内容
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
                # 以 UTF-8 编码重新保存
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                converted_files.append(file_path)
            except Exception as e:
                skipped_files.append((file_path, str(e)))

# ✅ 打印总结
print(f"✅ 成功转换文件数：{len(converted_files)}")
print(f"⚠️ 跳过文件数：{len(skipped_files)}")

if skipped_files:
    print("\n⚠️ 以下文件转换失败：")
    for path, error in skipped_files:
        print(f" - {path}：{error}")
