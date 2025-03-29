import cv2
import numpy as np
from mtcnn import mtcnn

img = cv2.imread('./img/timg.jpg')

model = mtcnn(weight_path1='./model_data/pnet.h5',
              weight_path2='./model_data/rnet.h5',
              weight_path3='./model_data/onet.h5')
threshold = [0.5, 0.6, 0.7]
rectangles = model.detectFace(img, threshold)

draw = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])
        paddingH = 0.01 * W
        paddingW = 0.02 * H
        crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                   int(rectangle[0] + paddingW):int(rectangle[2] - paddingW)]

        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        # 图像上绘制人脸矩形框
        cv2.rectangle(draw,  # 目标图像（副本）
                      (int(rectangle[0]), int(rectangle[1])),  # 矩形左上角坐标 (x1, y1)
                      (int(rectangle[2]), int(rectangle[3])),  # 矩形右下角坐标 (x2, y2)
                      (255, 0, 0),  # # 颜色（BGR格式，此处为蓝色）  OpenCV 使用 BGR 格式
                      1)  # 线宽（像素）

        for i in range(5, 15, 2):
            cv2.circle(draw,  # 目标图像（副本）
                       (int(rectangle[i + 0]), int(rectangle[i + 1])),  # 圆心坐标 (x, y)
                       2,  # 半径（像素）
                       (0, 255, 0))   # 颜色（BGR格式，此处为绿色） OpenCV 使用 BGR 格式

cv2.imwrite('./img/out.jpg', draw)

cv2.imshow('test', draw)
cv2.waitKey(0)
