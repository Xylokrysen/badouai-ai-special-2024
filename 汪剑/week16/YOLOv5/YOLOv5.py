import cv2
import torch

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5','yolov5s')

img = cv2.imread('street.jpg')

# 推理
result = model(img)

output_img = cv2.resize(result.render()[0],(512,512))

cv2.imshow('output_img: ',output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
