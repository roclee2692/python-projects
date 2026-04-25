def calculate_total(numbers):
    """计算列表中正数的和，并统计负数个数"""
    total = 0
    negative_count = 0

    for num in numbers:
        if num > 0:
            total += num
        elif num < 0:
            negative_count += 1
        else:
            print("遇到了 0，跳过处理")  # 断点可以打在这里

    return total, negative_count


# 测试用例
if __name__ == "__main__":
    data = [10, -5, 3, 0, -2, 7, -8]
    sum_result, neg_count = calculate_total(data)
    print(f"正数的和：{sum_result}")
    print(f"负数的个数：{neg_count}")