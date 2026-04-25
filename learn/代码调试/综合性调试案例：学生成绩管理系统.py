def calculate_average(scores):
    """计算平均分"""
    total = 0
    count = 0
    for score in scores:
        if score >= 0:  # 只统计有效成绩
            total += score
            count += 1
    average = total / count  # Bug 1：空列表时会除零
    return average


def find_max_score(scores):
    """找出最高分"""
    if not scores:
        return None
    max_score = scores[0]
    # Bug 2：循环范围错了，漏掉最后一个元素
    for i in range(1, len(scores) - 1):
        if scores[i] > max_score:
            max_score = scores[i]
    return max_score


def sort_scores(scores):
    """按成绩从高到低排序（冒泡排序）"""
    n = len(scores)
    for i in range(n):
        # Bug 3：内层循环范围错了，导致索引越界
        for j in range(0, n - i):
            if scores[j] < scores[j + 1]:
                scores[j], scores[j + 1] = scores[j + 1], scores[j]
    return scores


def main():
    # 测试数据
    student_scores = [85, 92, 78, -10, 90, 88]  # -10 是无效成绩
    empty_scores = []  # 空列表测试

    print("原始成绩:", student_scores)

    # 计算平均分
    avg = calculate_average(student_scores)
    print("平均分:", avg)

    # 找出最高分
    max_score = find_max_score(student_scores)
    print("最高分:", max_score)

    # 排序成绩
    sorted_scores = sort_scores(student_scores.copy())
    print("排序后成绩:", sorted_scores)

    # 测试空列表（会触发 Bug 1）
    print("\n测试空列表:")
    empty_avg = calculate_average(empty_scores)
    print("空列表平均分:", empty_avg)


if __name__ == "__main__":
    main()