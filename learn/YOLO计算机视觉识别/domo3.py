import cv2
import numpy as np
from ultralytics import YOLO
import math

print("\033[106m" + "正在检测飞鸟，降低飞机与飞鸟相撞的概率!!!!!!" + "\033[0m")

# 飞鸟的类别（假设飞鸟是classNames中的一项）
classNames = ["人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船",
              "交通信号灯", "消防栓", "禁止标记", "停车计时器", "长凳", "鸟", "猫",
              "狗", "房子", "羊", "奶牛", "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞",
              "手提包", "领带", "手提箱", "飞盘", "滑雪板", "滑雪单板", "运动秋", "风筝", "棒球棒",
              "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子", "酒杯", "杯子",
              "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花",
              "胡萝卜", "热狗", "披萨", "甜甜圈", "饼干", "椅子", "沙发", "盆栽植物", "床",
              "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机",
              "微波炉", "烤炉", "烤面包炉","钉子", "冰箱", "书", "钟", "花瓶", "剪刀",
              "泰迪熊", "吹风机", "牙刷"]

# 颜色
classColor = [
    (193, 182, 255), (214, 112, 218), (180, 105, 255), (237, 149, 100),
    (222, 188, 176), (255, 191, 0), (255, 255, 0), (170, 255, 127),
    (127, 255, 0), (0, 255, 0), (0, 255, 127), (210, 250, 250),
    (181, 228, 255), (173, 222, 255), (0, 140, 255), (112, 164, 244),
    (122, 160, 255), (80, 127, 255), (0, 69, 255), (71, 99, 255),
    (130, 0, 75), (226, 46, 138), (219, 112, 147), (238, 104, 123),
    (205, 92, 106), (139, 61, 72), (255, 0, 255), (255, 0, 255),
    (238, 130, 238), (255, 0, 255), (180, 206, 70), (235, 245, 135),
    (230, 224, 176), (160, 209, 95), (255, 255, 240), (255, 255, 225),
    (238, 238, 175), (231, 242, 212), (209, 206, 0), (139, 139, 0)
]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # 提高分辨率
cap.set(4, 720)

# 使用大模型，提高准确度（根据需要选择不同的模型）
model = YOLO("D:\\文档\\Python模型\\yolov8l.pt")  # 使用更大、更精确的模型

# 使用死循环进行不间歇读取
while True:
    ret, img = cap.read()

    # 图像预处理（例如，归一化处理等）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 进行实时检测
    results = model(img, stream=True)

    for r in results:  # 循环所有检测到的目标
        print("\033[96m" + "本次检测到 " + str(len(r.boxes.cls)) + " 个目标" + "\033[0m")

        for i in range(len(r.boxes)):
            cls = int(r.boxes[i].cls[0])
            conf = r.boxes[i].conf[0]
            if conf < 0.5:  # 过滤低置信度的检测框
                continue

            # 如果检测到的是“鸟”
            if classNames[cls] == "鸟":
                mask_np = r.masks.data[i].cpu().numpy()
                color = classColor[cls]
                alpha = 0.6  # 透明度

                # 创建一个与原图大小相同的透明overlay图层
                overlay = np.zeros_like(img)

                # 分别对每个通道应用掩码和颜色
                for c in range(3):
                    overlay[:, :, c][mask_np == 1] = (color[c] * alpha + img[:, :, c][mask_np == 1] * (1 - alpha)).astype(np.uint8)

                # 将overlay中的掩码区域合并到原图的相应位置
                img[mask_np == 1] = overlay[mask_np == 1]

                # 绘制边界框和类别信息
                x1, y1, x2, y2 = [int(x) for x in r.boxes[i].xyxy[0]]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{classNames[cls]} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示图像
    cv2.imshow('飞鸟检测 - 实时监控', img)

    # 按'q'键退出
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
