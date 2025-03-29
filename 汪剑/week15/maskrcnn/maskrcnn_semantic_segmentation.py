import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import ImageDraw, Image
import numpy as np
import cv2

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 选择是在 GPU 或者 CPU 上进行训练和推理
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


# 预处理
def preprocess_image(image):
    tranform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return tranform(image).unsqueeze(0)


# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

# 显示结果
def show_result(prediction):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    color_mapping = {
        1:(255,0,0),  # 人用蓝色表示
        2:(0,255,0),  # 自行车用绿色表示
        3:(0,0,255)   # 汽车用红色表示
    }

    for pred in prediction:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for mask,label,score in zip(masks,labels,scores):
            if score > 0.5:
                mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                # 根据 label 颜色映射，如果不在 color_mapping 字典中，默认颜色为白色 (255,255,255)
                '''
                零维张量是 PyTorch 处理数值的基础单元，相当于 NumPy 或 Python 里的单个数！
                标量（零维） 如：shape = torch.Size([])    维度 dim = 0  torch.tensor(5) 一个零维张量只是一个单独的数
                向量（一维） 如：shape = torch.Size([3])   维度 dim = 1  torch.tensor([1, 2, 3])
                矩阵（二维） 如：shape = torch.Size([2,3]) 维度 dim = 2 torch.tensor([[1,2,3],[4,5,6]])
                只有零维张量可以调用 .item()
                '''
                color = color_mapping.get(label.item(),(255,255,255))
                contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image,contours,-1,color,2)
    image = cv2.resize(image,(700,700))
    cv2.imshow('Result',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = 'street.jpg'
    prediction = infer(image_path)
    # image = Image.open(image_path)
    image = show_result(prediction)

