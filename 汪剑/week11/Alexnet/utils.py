import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    # 读取图片，RGB
    img = mpimg.imread(path)
    # 将图片修剪成中心正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    return crop_img


'''
tensorflow 中常用的差值方法，用于调整图像大小，默认为 tf.image.ResizeMethod.BILINEAR（双线性插值）： 
BILINEAR: 双线性插值，适合连续图像（如照片）。
NEAREST_NEIGHBOR: 最近邻插值，适合离散图像（如分割掩码）。
BICUBIC: 双三次插值，精度更高但计算更慢。
AREA: 面积插值，适合缩小图像时使用
'''


def resize_image(image,
                 size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False  # 布尔值，指定是否对齐图像角点。True: 调整大小时，源图像和目标图像的角点对齐；False: 默认设置，更常见
                 ):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)  # 在输入图像 image 的第 0 维（batch）添加一个维度，变为 4D 张量
        image = tf.image.resize_images(image, size, method, align_corners)

        # 重新设置图像形状以确保输出的形状固定为 [h, w, c]
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))  # -1: 表示动态批次维度（通常为 1）
        return image


def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]  # strip()：去掉每行字符串的首尾空白字符（如空格、换行符）
    '''
    将概率从大到小排列的结果的序号存入 pred
    np.argsort(prob)：返回数组 prob（概率数组） 中的值从小到大排序的索引[::-1]: 反转排序后的索引，使其从大到小排列
    例如：[0.1, 0.7, 0.2] → [1, 2, 0]
    '''
    pred = np.argsort(prob)[::-1]

    # 取最大的1个、5个
    top1 = synset[pred[0]]  # 获取概率最高的类别（Top 1）
    print(('Top1: ', top1, prob[pred[0]]))  # top1：类别名称。 prob[pred[0]]：最大概率值
    top5 = [(synset[pred[i], prob[pred[i]]]) for i in range(5)]
    print('Top5: ', top5)
    return top1
