import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np

'''
import torch         一个用于深度学习的开源机器学习库，提供张量计算和自动梯度计算等功能
import torchvision   PyTorch 生态系统中的计算机视觉库，提供数据集、预训练模型和图像转换等功能

from torchvision.models.detection import fasterrcnn_resnet50_fpn
从 torchvision.models.detection 子模块中导入 Faster R-CNN 预训练模型，
具体是 Faster R-CNN 结合 ResNet-50 主干网络 (Backbone network) 和 Feature Pyramid Network (FPN)，用于目标检测

from torchvision.transforms import functional as F
从 torchvision.transforms 模块中导入 functional 并重命名为 F，提供一系列用于图像处理的函数，如 resize、normalize、to_tensor 等

from PIL import Image,ImageDraw
从 PIL（Python Imaging Library，现为 Pillow）中导入 Image（用于加载和处理图片）和 ImageDraw（用于在图片上绘制图形）
'''

# 加载预训练模型
# fasterrcnn_resnet50_fpn 这个模型是在 COCO 数据集（包含 80 类物体）上训练好的，可以直接用于推理
model = fasterrcnn_resnet50_fpn(pretrained=True)  # pretrained=True 代表 使用在 COCO 数据集上训练好的权重
model.eval()  # 将模型设为 评估模式（evaluation mode），避免 BatchNorm 和 Dropout 影响推理结果

# 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

'''
Compose 用于 将多个变换操作组合在一起，形成一个数据预处理管道。
传入的参数是一个 变换列表，会按照 顺序 应用

使用 Compose 的好处是：
方便扩展：如果以后需要增加其他变换（比如 Resize()、Normalize()），可以直接往 Compose 里加，而不用修改代码结构
代码统一：在 PyTorch 训练中，数据预处理通常都封装在 Compose 里，这样可以保持代码一致性

实例：
transform = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])
'''


# 加载图像并进行预处理
def preprocess_image(image):
    # 将 PIL.Image 转换为 张量（Tensor）ToTensor 默认归一化到 [0,1]，通道顺序变为 C×H×W（通道数 × 高 × 宽）
    tranform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    return tranform(image).unsqueeze(0)  # 添加 batch 维度 [1, C, H, W]


# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    '''
    prediction 是一个 list，其中每个元素都是一个 字典，包含：
    boxes: 预测的目标边界框坐标（[x_min, y_min, x_max, y_max]）
    labels: 目标类别索引（与 COCO 数据集类别对应）
    scores: 置信度分数（越高表示模型越确信检测结果）

    比如：
    [
        {
            'boxes': tensor([[x1, y1, x2, y2], [x1, y1, x2, y2], ...]),  # 目标框坐标
            'labels': tensor([label1, label2, ...]),  # 目标类别索引
            'scores': tensor([score1, score2, ...])  # 置信度
        },
    ]  PyTorch 目标检测模型的输出是一个列表，其中每个元素对应一张输入图像的检测结果
    '''
    return prediction


# 显示结果
def show_result(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()  # prediction[0] 表示第一张图片
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # 阈值根据需要调整
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')
    image.show()


image_path = 'street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)
