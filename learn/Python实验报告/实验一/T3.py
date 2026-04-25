# 题目3：通过数字查询对应的星期几
# 描述：输入1-7的数字，输出对应的中文星期名称

try:
    week_num = int(input("please input number 1-7: "))
    
    # 星期列表，索引0为空字符串（占位），1-7分别对应星期一到星期日
    week_list = ["", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    
    # 验证输入范围是否有效
    if 1 <= week_num <= 7:
        print("---- T3 Result ----")
        print(f"input number {week_num} reflect: {week_list[week_num]}")
    else:
        print("error: please input number between 1-7")
        
except ValueError:
    print("error: invalid input, please input an integer")
