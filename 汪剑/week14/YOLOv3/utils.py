import _json
import numpy as np
import tensorflow as tf
from PIL import Image
from _collections import defaultdict

'''
model.load_weights() 是一种便捷的方式，适用于格式为 TensorFlow/Keras 的预训练权重文件，
但无法直接处理非 TensorFlow 格式的权重文件（如 Darknet 格式的权重），需要自定义函数处理

从一个二进制的权重文件 (weights_file) 中加载 Darknet53 模型的权重数据，并将这些数据赋值给相应的 TensorFlow 变量（即模型中的权重和偏置）。
函数返回一个 assign_ops，这是一个赋值操作列表，用于 TensorFlow 会话中执行赋值操作。
var_list: 一个包含 TensorFlow 变量（权重、偏置等）的列表，这些变量需要被赋值。
weights_file: 包含预训练权重的文件，通常是二进制格式

举例:
var_list = [
    # 第一个卷积层（conv1）的核权重
    <tf.Variable 'conv2d_1/kernel:0' shape=(3, 3, 3, 32) dtype=float32>,  # [h, w, in_c, out_c]
    
    # 第一个卷积层（conv1）的偏置
    <tf.Variable 'conv2d_1/bias:0' shape=(32,) dtype=float32> # 偏置，可能有也可能没有
    
    # 第一个批量归一化层（bn1）的参数
    <tf.Variable 'batch_normalization_1/beta:0' shape=(32,) dtype=float32>,
    <tf.Variable 'batch_normalization_1/gamma:0' shape=(32,) dtype=float32>,
    <tf.Variable 'batch_normalization_1/moving_mean:0' shape=(32,) dtype=float32>,
    <tf.Variable 'batch_normalization_1/moving_variance:0' shape=(32,) dtype=float32>,
    
    # 第二个卷积层（conv2）的核权重和偏置
    <tf.Variable 'conv2d_2/kernel:0' shape=(3, 3, 32, 64) dtype=float32>,
    <tf.Variable 'conv2d_2/bias:0' shape=(64,) dtype=float32>,
    
    # 后续其他层...
]

weights = [
    # Conv2D kernel (3x3, 3 input channels, 2 output channels)
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    # Conv2D bias (1 bias for each output channel)
    0.5, 0.5,
    # BatchNorm gamma (scale factors)
    0.6, 0.7,
    # BatchNorm beta (shift factors)
    0.8, 0.9,
    # BatchNorm mean
    0.1, 0.2,
    # BatchNorm variance
    0.3, 0.4
]
'''

# 这里的 darknet53.weights 权重文件对应的顺序是：bn参数或者偏置参数在前，卷积核参数在后
def load_weights(var_list, weights_file):
    '''
    Introduction
    ------------
        加载预训练好的darknet53权重文件
    Parameters
    ----------
    :param var_list: 赋值变量名
    :param weight_file: 权重文件
    :return: assign_ops: 赋值更新操作
    '''

    # 以二进制读取模式 (rb) 打开
    with open(weights_file, 'rb') as fp:
        '''
        读取文件的前 5 个 32 位整数，通常这些整数表示模型的元数据，比如模型的尺寸等信息（这些数据通常不直接用于变量赋值，因此被忽略，使用 _ 来丢弃）
        '''
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        '''
        读取权重文件中剩余的所有权重数据，并将其存储为一维浮点数组 weights
        '''
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0  # 这是一个指针，用于在 weights 数组中追踪已经读取的权重位置
    i = 0
    assign_ops = []
    # 每次迭代，处理当前层的权重和相关的 Batch Normalization 层（如果有的话）
    while i < len(var_list) - 1:
        var1 = var_list[i]  # 如 conv2d_1/kernel
        var2 = var_list[i + 1]  # 如 batch_normalization1/gamma
        if 'conv2d' in var1.name.split('/')[-2]:
            # 如果下一层是 BN 层
            if 'batch_normalization' in var2.name.split('/')[-2]:
                '''
                gama,beta,mean,var 分别对应如下：
                gama（缩放项）
                beta（偏置项）
                moving_mean（均值）
                moving_variance（方差）
                '''
                gama, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gama, mean, var]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()  # 如将 (32,) 转成 [32]
                    num_params = np.prod(shape)  # 32
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)  # 提取相应的权重并重塑
                    ptr += num_params
                    # # validate_shape=True：启用形状验证，确保赋值的数据形状与目标变量的形状一致。如果不一致会报错
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                i += 4  # 因为有 gama,beta,mean,var 4个变量

            elif 'conv2d' in var2.name.split('/')[-2]:
                bias = var2  # 偏置
                bias_shape = var2.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                i += 1

            shape = var1.shape.as_list()  # 如 (3, 3, 3, 32) 转成 [3, 3, 3, 32] 即为 [h, w, in_c, out_num]
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))  # (out_num,in_c,h,w)
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))  # (h,w,in_c,out_num)
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))

            i += 1

    return assign_ops


def letterbox_image(image, size):
    '''
    Introduction
    ------------
        对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    Parameters
    ------------
    :param image: 输入图像，一个 PIL 图像对象
    :param size: 目标图像大小
    :return: box_image: 缩放后的图像
    '''

    #image = Image.open(image)
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    # Image.new() 创建一个新的图像，大小为 size（即目标宽度 w 和目标高度 h），填充颜色是 (128, 128, 128)，这是灰色。
    # 这个新的图像将作为背景，用来容纳缩放后的图像
    boxed_image = Image.new('RGB', size, (128, 128, 128))

    # boxed_image.paste() 方法将缩放后的图像 resized_image 粘贴到 boxed_image 中，位置是 居中的。
    # 通过 (w-new_w)//2 和 (h-new_h)//2 计算出粘贴的起始位置，这样就能保证缩放后的图像在目标图像中居中对齐
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))

    return boxed_image


def draw_box(image, bbox):
    '''
    Introduction
    ------------
        通过tensorboard把训练数据可视化
    Parameters
    ----------
    :param image: 训练数据图片
    :param bbox: 训练数据图片中标记 box 坐标，一个包含边界框坐标的 TensorFlow 张量，形状为 (batch_size, 1, 5)
                 1 表示每个图像只有一个标注框（这个维度可以在需要时扩展） (n, m, 5)
    :return:
    '''

    # num_or_size_splits = 5：表示将 bbox 张量沿第 2 维（即 axis=2）分割为 5 个元素，即：xmin, ymin, xmax, ymax, label
    xmin, ymin, xmax, ymax, label = tf.split(value=bbox, num_or_size_splits=5, axis=2)

    # 图像维度的张量：(batch_size, height, width, channels)
    height = tf.cast(tf.shape(image)[1], tf.float32)
    width = tf.cast(tf.shape(image)[0], tf.float32)

    # 归一化坐标
    new_bbox = tf.concat(
        [tf.cast(ymin, tf.float32) / height,  # 将 ymin 归一化为相对于图像高度的比例
         tf.cast(xmin, tf.float32) / width,  # 将 xmin 归一化为相对于图像宽度的比例
         tf.cast(ymax, tf.float32) / height,  # 将 ymax 归一化为相对于图像高度的比例
         tf.cast(xmax, tf.float32) / width],  # 将 xmax 归一化为相对于图像宽度的比例
        2)  # 按照第 2 维(即 axis = 2) 拼接成一个新的张量 new_bbox，其形状为 (N, 1, 4)，表示每个框的归一化坐标

    new_image = tf.image.draw_bounding_boxes(image, new_bbox)

    # 将绘制了标注框的图像 new_image 添加到 TensorBoard 中。它会在 TensorBoard 中显示出当前图像以及它的标注框
    tf.summary.image('input', new_image)


'''
这个函数的作用是计算 VOC（Visual Object Classes） 测试中用于评估目标检测模型性能的 AP（Average Precision） 值。
VOC 是一个标准的计算机视觉挑战赛，它包括多种任务，如目标检测、图像分类等。
对于目标检测任务，AP 是评估检测性能的常用指标，表示模型在不同召回率下的精度
'''


def voc_ap(rec, prec):
    '''

    :param rec: 召回率（Recall）列表，表示在每个预测阈值下的召回率。召回率是检测到的正样本占所有正样本的比例
    :param prec: 精度（Precision）列表，表示在每个预测阈值下的精度。精度是检测到的正样本占所有检测样本的比例
    :return:
    '''

    # 对召回率列表 rec 增加边界值 0 和 1。因为计算 AP 时需要从 0 到 1 的连续区间来表示召回率的变化，所以要在开头插入 0，在末尾插入 1
    rec.insert(0, 0.0)  # 在列表开头插入 0.0
    rec.append(1.0)  # 在列表末尾插入 1.0
    mrec = rec[:]

    # 在精度列表开头和末尾分别插入 0
    prec.insert(0,0.0)
    prec.append(0.0)

    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i],mpre[i+1])

    i_list = []
    # 找出所有不同的召回率点，即找到那些 召回率发生变化的索引
    for i in range(1,len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        # 对于每个不同的召回率区间，计算精度和召回率的差值，并与对应的精度值相乘
        ap += ((mrec[i] - mrec[i-1]) * mpre[i])
    return ap,mrec,mpre
