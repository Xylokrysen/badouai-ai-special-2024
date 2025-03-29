import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 选择在 GPU 上训练还是 CPU 上训练
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


# 加载图像并进行预处理
def preprocess_image(image):
    tranform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return tranform(image).unsqueeze(0)  # 添加 bathc_size 维度


# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction


'''
最终输出被组织为字典列表（每个图像对应一个字典）：
outputs = [
    {
        'boxes':   tensor([[x1, y1, x2, y2], ...]),  # 经过NMS和阈值过滤的边界框
        'labels':  tensor([class_id1, class_id2, ...]),  # 类别ID
        'scores':  tensor([score1, score2, ...]),       # 置信度分数
        'masks':   tensor([[[mask_probabilities]]])     # 二值化前的掩码概率图
    }
]

其中 masks 形状为 [N,1,H,W]
N: 表示当前图像中检测到的 实例数量（即检测到的目标数量）
1: 表示每个实例的掩码是 单通道的二进制概率图
   掩码的每个像素值在 [0, 1] 之间，表示该像素属于目标实例的概率
   通常通过阈值（如 0.5）二值化，得到 0（背景）或 1（目标）的掩码
H 和 W: 表示掩码的 高度（Height） 和 宽度（Width），通常与输入模型的图像尺寸一致
'''


# 显示结果
def show_result(image, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 创建一个空字典 instance_colors，用于存储每个实例（检测到的对象）对应的随机颜色，这样后续绘制时每个对象都有独特的颜色
    instance_colors = {}

    for pred in predictions:
        boxes = pred['boxes'].cpu().numpy()  # 先移动到 CPU，再转换为 NumPy。Tensor 必须在 CPU 上才能转换为 NumPy 数组
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()

        # 此时 mask 的 形状为 (1,H,W)
        for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            if score > 0.5:
                mask = mask[0]  # 降维，形状变成 (H,W)
                # 对掩膜进行二值化处理
                # 使用 .astype(np.uint8) 将布尔数组转换为 8 位无符号整型数组（1 表示前景，0 表示背景）
                mask = (mask > 0.5).astype(np.uint8)
                if i not in instance_colors:
                    instance_colors[i] = (
                        np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                color = instance_colors[i]  # 获取当前颜色
                '''
                cv2.findContours() 是 OpenCV 中用于检测二值化图像中的轮廓的函数，适用于对象分割、形状分析等任务
                contours, hierarchy = cv2.findContours(image, mode, method)
                其中：
                mode 轮廓检索模式：
                cv2.RETR_EXTERNAL：只检测最外层轮廓，内部轮廓被忽略
                cv2.RETR_LIST：检测所有轮廓，但不建立层次关系
                cv2.RETR_CCOMP：轮廓分为两个级别：外部边界和内部边界
                cv2.RETR_TREE：检测所有轮廓，并建立完整的层级结构
                
                method 轮廓逼近方法：
                cv2.CHAIN_APPROX_NONE：保留所有轮廓上的点
                cv2.CHAIN_APPROX_SIMPLE：只保留轮廓的关键点，减少存储量
                cv2.CHAIN_APPROX_TC89_L1：使用 Teh-Chin 链码近似轮廓
                cv2.CHAIN_APPROX_TC89_KCOS：使用 Teh-Chin K 余弦近似轮廓
                
                返回值：
                contours：一个 Python 列表，每个元素是一个轮廓（NumPy 数组），表示检测到的对象的边界点集合
                hierarchy：一个 NumPy 数组，包含轮廓的层次关系（父轮廓、子轮廓等）
                '''
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                '''
                cv2.drawContours() 是 OpenCV 中用于在图像上绘制轮廓的函数，通常与 cv2.findContours() 结合使用，用于可视化检测到的物体边界
                cv2.drawContours(image, contours, contourIdx, color, thickness)
                其中：
                image：要绘制轮廓的目标图像（应为彩色图 BGR）。
                contours：轮廓列表（通常由 cv2.findContours() 生成）。
                contourIdx：设为 -1 时，绘制所有轮廓。设为 0, 1, 2, ... 时，绘制特定索引的轮廓。
                color：轮廓颜色，格式为 (B, G, R)，例如 (0, 255, 0) 代表绿色。
                thickness：设为 >0 时，表示轮廓线的宽度（单位：像素）。设为 -1 时，表示填充整个轮廓区域
                
                返回值：
                该函数直接修改 image，不返回新图像
                '''
                cv2.drawContours(image, contours, -1, color, 2)

                top_left = (int(box[0]), int(box[1]))
                bottom_right = (int(box[2]), int(box[3]))
                cv2.rectangle(image, top_left, bottom_right, color, 2)
                '''
                cv2.putText() 语法
                cv2.putText(image, text, org, fontFace, fontScale, color, thickness, lineType)
                参数说明：
                1、image：目标图像（NumPy 数组）。
                2、text：要绘制的文本字符串。
                3、org：文本起始位置（左下角坐标），格式 (x, y)。
                4、fontFace：字体类型，如：cv2.FONT_HERSHEY_SIMPLEX（常用） cv2.FONT_HERSHEY_COMPLEX
                5、fontScale：字体缩放因子（数值越大，字体越大）。
                6、color：文本颜色，格式 (B, G, R)，如 (0, 0, 255) 表示红色。
                7、thickness：文本线条粗细。
                8、lineType：线条类型（可选），常用：cv2.LINE_AA（抗锯齿，常用） cv2.LINE_4（4 连接） cv2.LINE_8（8 连接，默认）
                '''
                cv2.putText(image, str(label) + ' ' + str(round(score,2)), (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.namedWindow('Result', cv2.WINDOW_NORMAL)  # 允许窗口缩放
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 示例
image_path = 'street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)
