import sys
from operator import itemgetter
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ------------------------#
#    计算原始输入图像
#    每一次缩放的比例
# ------------------------#
def calculateScales(img):
    copy_img = img.copy()
    pr_scale = 1.0
    h, w, _ = copy_img.shape

    # 引申优化项 = resize(h*500/min(h,w),w*500/min(h,w))
    if min(w, h) > 500:
        pr_scale = 500.0 / min(w, h)
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(w, h)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    scales = []
    factor = 0.709  # sqrt(2)/2
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales


# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    '''
    :param rectangles: 左上和右下坐标 (xmin,ymin,xmax,ymax)
    :return:
    '''
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    # np.maximum(w, h) 对于每个矩形取宽度和高度中的较大值，作为新的正方形的边长
    l = np.maximum(w, h).T

    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5

    # np.repeat([l],2,axis=0).T 将一维数组 l 重复为两列，使得每个正方形的边长可以同时加到 x 和 y 坐标上
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T

    return rectangles


# -------------------------------------#
#   非极大抑制
# -------------------------------------#
def NMS(rectangles, threshold):
    '''
    :param rectangles: 列表或数组，每个元素为 [x1, y1, x2, y2, score]，分别代表候选框左上角、右下角坐标和对应的置信分数
    :param threshold:
    :return:
    '''
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    # 面积 = (宽度 + 1) * (高度 + 1)
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)  # 这里加 1 是为了确保边界像素也被计入（常见于目标检测中）
    I = np.array(s.argsort())  # 分数从低到高排序后的索引数组

    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # # I[-1]为当前最高分的框，其它候选框为I[0:-1]
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        # 计算交集面积
        inter = w * h

        # 计算交并比
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)

        pick.append(I[-1])

        I = I[np.where(o <= threshold)[0]]

    result_rectangle = boxes[pick].tolist()
    return result_rectangle


# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    '''
    :param cls_prob: 分类概率（人脸 vs 非人脸），置信度矩阵，表示每个位置预测为人脸的概率（shape: (H, W))
    :param roi: 每个候选框的边界框偏移量，偏移量矩阵，表示每个候选框的位置修正（shape: (4, H, W))
    :param out_side: 12-net 输出的特征图大小
    :param scale: 该金字塔层级相对于原图的缩放比例
    :param width: 原始图像的宽度
    :param height: 原始图像的高度
    :param threshold: 人脸判别的阈值（概率低于 threshold 的区域将被忽略）
    :return:
    '''
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)  # (4, H, W) 变成 (W,H,4)  交换轴，使数据格式统一

    '''
    目的：计算 候选框中心点间隔（大约为 2）
    由于 MTCNN 采用滑动窗口检测，每次滑动 stride 个像素:
    out_side = 5 → stride = (2×5 - 1) / (5-1) = 2.25
    '''
    stride = 0
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
    # 找到可能为人脸的点
    (x, y) = np.where(cls_prob >= threshold)

    # 候选框的索引位置
    boundingbox = np.array([x, y]).T

    # 找到对应原图的位置，计算候选框的左上角 (bb1) 和右下角 (bb2)
    bb1 = np.fix((stride * boundingbox + 0) * scale)
    bb2 = np.fix((stride * boundingbox + 11) * scale)  # np.fix() 取整，确保坐标是整数

    boundingbox = np.concatenate((bb1, bb2), axis=1)  # 得到每个候选框位置 (xmin,ymin,xmax,ymax)

    dx1 = roi[0][x, y]  # 左边界偏移量
    dx2 = roi[1][x, y]  # 上边界偏移量
    dx3 = roi[2][x, y]  # 右边界偏移量
    dx4 = roi[3][x, y]  # 下边界偏移量
    offset = np.array([dx1, dx2, dx3, dx4]).T

    # 计算最终调整后的 boundingbox
    boundingbox = boundingbox + offset * 12.0 * scale

    score = np.array([cls_prob[x, y]]).T

    # 将矩形调整为正方形
    rectangles = np.concatenate((boundingbox, score), axis=1)
    rectangles = rect2square(rectangles)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])

    return NMS(pick, 0.3)


# -------------------------------------#
#   对Rnet处理后的结果进行处理
# -------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    '''
    :param cls_prob: 一个二维数组，每一行表示一个候选框的类别概率（例如 [背景概率, 人脸概率]），形状为 (N,2) N表示候选框数量
    :param roi: 回归偏移量数组，形状一般为 (N, 4)，对应每个候选框的 [dx1, dx2, dx3, dx4]，用于微调框的位置
    :param rectangles: 候选框列表或数组，每个框初始形式为 [x1, y1, x2, y2]（可能还附带其他信息）
    :param width: 原始图像的宽度
    :param height: 原始图像的高度
    :param threshold: 人脸检测阈值
    :return:
    '''
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    # 根据筛选结果提取候选框坐标
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 计算每个候选框的宽度和高度（未修正前）
    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate([x1, y1, x2, y2, sc], axis=1)
    rectangles = rect2square(rectangles)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)


# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    '''
    :param cls_prob: 类别概率数组，形状通常为 (N, 2)，其中每一行表示 [背景概率, 人脸概率]。我们主要取第二列（索引 1）的概率作为人脸置信度
    :param roi: 回归偏移量数组，形状为 (N, 4)，每一行包含候选框的四个修正系数 [dx1, dx2, dx3, dx4]，分别对应左、上、右、下边界的调整比例
    :param pts: 人脸关键点（landmarks）预测数组，形状为 (N, 10)，每一行包含 10 个数值，表示 5 个关键点的相对坐标。按照该代码的使用顺序
                第 0 个、1 个、2 个、3 个、4 个值分别表示 5 个关键点的 x 方向相对坐标；
                第 5 个、6 个、7 个、8 个、9 个值分别表示对应关键点的 y 方向相对坐标
    :param rectangles: 候选人脸框列表或数组，每个框为 [x1, y1, x2, y2]
    :param width: 原始图像的宽度
    :param height: 原始图像的高度
    :param threshold: 人脸检测阈值
    :return:
    '''
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    # 提取回归偏移量
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    # 计算人脸关键点在原图上的绝对坐标
    # 对于每个候选框，pts[pick, i] 表示关键点相对坐标（归一化到 [0,1]）
    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    # 根据回归偏移量修正候选框
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    # 拼接修正后候选框与关键点信息
    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2,
                        rectangles[i][4],
                        rectangles[i][5], rectangles[i][6],
                        rectangles[i][7], rectangles[i][8],
                        rectangles[i][9], rectangles[i][10],
                        rectangles[i][11], rectangles[i][12],
                        rectangles[i][13], rectangles[i][14]])
    return NMS(pick, 0.3)
