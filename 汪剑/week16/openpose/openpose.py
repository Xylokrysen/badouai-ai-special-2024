import cv2
import torch
import torchvision
import numpy as np

# 加载预训练模型
model = torch.hub.load('CMU-Visual-Computing-Lab/openpose', 'pose_resnet50', pretrained=True)
model.eval()

# 图像预处理
def preprocess_image(image):
    tranform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return tranform(image).unsqueeze(0)

# 读取图像
image_path = 'demo.jpg'
image = cv2.imread(image_path)
image_tensor = preprocess_image(image)

# 推理和结果处理
with torch.no_grad():
    output = model(image_tensor)
# 表示关节点热力图
'''
heatmaps 形状为 (C,H,W)
C：关节点的类别数（如COCO数据集的18个关键点或Body-25模型的25个关键点）
H 和 W：热力图的高度和宽度
'''
heartmap = output[0].cpu().numpy()
keypoints = np.argmax(heartmap,axis=0)
for i in range(heartmap.shape[0]):  #  heartmap.shape[0] = C
    y,x = np.unravel_index(np.argmax[heartmap[i]],heartmap[i].shape)
    cv2.circle(image,(x,y),5,(0,255,0),-1)

cv2.imshow('Keypoints: ',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
