def find_average(scores):
    """计算平均分（有 Bug！）"""
    total = 0
    count = 0
    for score in scores:
        if score >= 0:
            total += score
            count += 1
    # 这里有个隐藏问题，你能通过调试发现吗？
    average = total / count
    return average


# 测试用例
if __name__ == "__main__":
    # 正常用例
    scores1 = [85, 92, 78, 90]
    print(f"正常平均分：{find_average(scores1)}")

    # 异常用例（会触发 Bug）
    scores2 = []  # 空列表
    print(f"空列表平均分：{find_average(scores2)}")