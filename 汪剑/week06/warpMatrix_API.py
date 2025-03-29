import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
result3 = img.copy()

# src 和 dst 对应的顶点坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]]) # 创建一个 4×2 的 NumPy 数组，数据类型为 float32
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print('原图尺寸：',img.shape)
# 生成透视变换矩阵，进行透视变换
# Perspective Transformation  透视变换
m = cv2.getPerspectiveTransform(src,dst)
print('warpMatrix:\n',m)
result = cv2.warpPerspective(result3,m,(337,448))
cv2.imshow('src:',img)
cv2.imshow('dst:',result)
cv2.waitKey(0)  # 等待用户按键，参数 0 表示无限等待，直到用户按下键盘任意键
cv2.destroyAllWindows() # 关闭所有 OpenCV 创建的窗口，释放资源
