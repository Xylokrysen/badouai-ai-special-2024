import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from YOLOv3.model.yolo3_model import yolo


class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        '''
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
        :param obj_threshold: 目标检测维物体的阈值
        :param nms_threshold: nms阈值
        :param classes_file: 类别文件
        :param anchor_file: 锚框文件
        '''

        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        # 预读取
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        # 读取种类名称
        self.class_names = self._get_class()
        # 读取先验框
        self.anchors = self._get_anchors()

        # 画框框用
        # 生成每个类别的 HSV 色值，然后根据这个色值生成对应的 RGB 颜色
        # hsv_tuples 列表的每个元素是一个 HSV 元组，表示类别 x 对应的色相、饱和度和明度（如 (0, 1, 1)、(0.1, 1, 1) 等等）
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        # 将 HSV 色值转换为 RGB 色值
        '''
        map(): 是一个 Python 内置函数，用于将一个函数应用到一个可迭代对象的每个元素上，并返回一个迭代器
               例如，map(func, iterable) 会对 iterable 中的每个元素应用 func 函数
        colorsys.hsv_to_rgb(): 是一个用于将 HSV（色相、饱和度、明度） 色值转换为 RGB（红色、绿色、蓝色） 色值的函数
                               这个函数的输入是三个值：色相（H）、饱和度（S）、明度（V）。它返回一个表示 RGB 的元组（每个值在 0 到 1 的范围内）
        *x: 表示将 x 中的值解包传递给 hsv_to_rgb 函数。假设 x 是一个包含 3 个元素的元组（例如 (H, S, V)），
            *x 会将这 3 个元素拆开，分别传递给 hsv_to_rgb()
        '''
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 将 RGB 颜色值从 [0, 1] 范围转换为 [0, 255] 范围，即将浮动值转换为整数（适用于图像处理）
        self.colors = list(map(lambda x: (int(x[0]*255),int(x[1]*255),int(x[2]*255)),self.colors))

        random.seed(10101)  # 设置随机种子，以确保接下来的随机操作是可重复的
        random.shuffle(self.colors)
        random.seed(None)


    def _get_class(self):
        '''
         Introduction
        ------------
            读取类别名称
        ------------
        :return: class_names: 类别名称列表
        '''

        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        '''
        Introduction
        ------------
            读取anchors数据
        ------------
        :return:
        '''

        anchor_path = os.path.expanduser(self.anchors_path)
        with open(anchor_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    '''
    feats: 是 YOLO 模型的最后一层的输出特征图，通常具有形状 [batch_size, grid_size_y, grid_size_x, num_anchors * (5 + num_classes)]，
           其中 5 + num_classes 表示每个网格单元的 5 个值（x, y, w, h, confidence）加上类别的数量
    anchors: 是模型中预定义的锚框（anchors），它们的作用是帮助网络预测边界框的形状。每个锚框对应于一个候选框的宽度和高度
    num_classes: 类别的数量（如对于 COCO 数据集，num_classes = 80；VOC数据集，num_classes = 20）    
    '''

    # 其实是解码过程
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        '''
        Introduction
        ------------
            根据yolo最后一层的输出确定 bounding box
        Parameters
        ----------
        :param feats: yolo 模型最后一层输出
        :param anchors: anchors 的位置，形状为 (num_anchors,2)
        :param num_classes: 类别数量
        :param input_shape: 输入大小
        :return: box_xy: 每个边界框的中心坐标（x, y），归一化到 [0, 1] 范围内
                 box_wh: 每个边界框的宽度和高度，归一化到 [0, 1] 范围内
                 box_confidence: 每个边界框的置信度，表示该框内是否存在目标
                 box_class_probs: 每个边界框属于各个类别的概率
        '''
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])  # 以便与特征图的大小进行广播

        # 获取 feature map 的网格大小
        grid_size = tf.shape(feats)[1:3]  # 即 (grid_size_y, grid_size_x)

        # 重新调整 feats 的形状
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 这里构建 13*13*1*2 的矩阵，对应每个格子加上对应的坐标
        '''
        假设网格形状是 (13,13)
        tf.range(grid_size[0]) 对应是 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  形状为 (13,)
        tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]) 形状为 (13,1,1,1)
        tf.tile(..., [1, grid_size[1], 1, 1])  形状为 (13,13,1,1)
        '''
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])  # 形状 (13,13,1,1)
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])  # 形状 (13,13,1,1)
        grid = tf.concat([grid_x, grid_y], axis=-1)  # 形状 (13,13,1,2)
        grid = tf.cast(grid, tf.float32)

        # 计算边界框的坐标
        # 将 x,y 坐标归一化，相对网格的位置
        '''
        tf.sigmoid(predictions[...,:2],grid): 将网格坐标和预测的中心点坐标相加，得到相对于整个图像的中心点坐标
        / tf.cast(grid_size[::-1], tf.float32): 对坐标进行归一化，除以 grid_size[::-1]，即特征图的宽和高，确保中心点坐标归一化到 [0, 1] 范围内
        '''
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)

        '''
        e^(predictions[..., 2:4]) * anchors_tensor: 预测的相对宽度和高度的比例与锚框的宽度和高度相乘得到预测的宽度和高度
        / tf.cast(input_shape[::-1], tf.float32): 对宽度和高度进行归一化，使它们的值在 [0, 1] 范围内，归一化到输入图像的大小
        '''
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)

        # 计算框的置信度和类别概率
        box_confidence = tf.sigmoid(predictions[..., 4:5])  # 每个边界框是否包含目标的概率
        box_class_probs = tf.sigmoid(predictions[..., 5:])  # 每个边界框中各个类别的概率

        return box_xy, box_wh, box_confidence, box_class_probs

    '''
        处理不同尺寸的图像时，尤其是目标检测模型（如 YOLO、Faster R-CNN 等），通常会遵循以下原则：
        目标图像尺寸：
        通常会将图像缩放到统一的尺寸，例如 416x416，640x640 等，这是为了方便模型训练和推理。
        为了确保图像保持原始长宽比（即不被拉伸或变形），常见的做法是 缩放后填充（padding），而不是直接拉伸或插值。

        在目标检测中，我们一般会按比例缩放图像，保持原始图像的长宽比。
        例如，假设原始图像是一个 800x600 的矩形图像，目标尺寸是 416x416，那么它会按比例缩放为 416x312 的图像（按宽度的比例进行缩放），
        然后再填充空白区域，填充的区域通常是 128, 128, 128（灰色）
        '''

    '''
    box_xy 和 box_wh 通常是归一化的尺寸，即它们是相对于输入图像的尺寸进行的归一化
    '''

    # 获得在原图上框的位置
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
        :param box_xy: 物体框左上角坐标，box_xy 形状通常为 (batch_size, num_boxes, 2)
        :param box_wh: 物体框的宽高，box_wh 形状通常为 (batch_size, num_boxes, 2)
        :param input_shape: 输入的大小
        :param imagh_shape: 图片的大小
        :return: boxes: 物体框的位置
        '''

        '''
        ... 和 ::-1 是 Python 切片（slice）操作的一部分:
        ... 是 Python 的切片操作符，它表示“所有的维度”或“保留所有的元素”
        ::-1 是 Python 中的一种切片语法，表示反转数组或列表的顺序

        box_xy[..., ::-1] 会选择张量的所有批次和所有框，只反转最后一个维度（x, y 的顺序）
        '''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        # 416,416
        input_shape = tf.cast(input_shape, dtype=tf.float32)  # tf.cast 将张量进行数据类型转换

        # 实际图片的大小
        image_shape = tf.cast(image_shape, dtype=tf.float32)  # 如 (800,600)

        # tf.reduce_min 取缩放比例最小值，这样可以确保图像按照长宽比的比例进行缩放，同时保持原始图像不会被拉伸变形
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))  # 如 (416,312)

        offset = (input_shape - new_shape) / 2.0 / input_shape  # 表示缩放后的图像需要偏移一定的量
        scale = input_shape / new_shape  # 表示图像的缩放比例

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([box_mins[..., 0:1],  # ymin 形状 (batch_size, num_boxes, 1)
                           box_mins[..., 1:2],  # xmin 形状 (batch_size, num_boxes, 1)
                           box_maxes[..., 0:1],  # ymax 形状 (batch_size, num_boxes, 1)
                           box_maxes[..., 1:2]],  # xmax 形状 (batch_size, num_boxes, 1)
                          axis=-1)  # axis=-1 表示按照最后一维度将张量进行拼接  形状 (batch_size, num_boxes, 4)

        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    # ---------------------------------------#
    #   对三个特征层解码
    #   获取预测框在原图上的位置以及类别置信度
    # ---------------------------------------#
    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        '''
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
        :param feats: yolo 输出的 feature map
        :param anchors: anchors 的位置，形状为 (num_anchors,2)
        :param classes_num: 类别数目
        :param input_shape: 输入大小
        :param image_shape: 图片大小
        :return: boxes: 物体框的位置
                 boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        '''
        # 获取特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)

        # 寻找在原图上的位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])

        # 获得类别置信度 box_confidence * box_class_probs
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        '''
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
        :param yolo_outputs:  yolo 模型输出。YOLO 模型的输出，通常是一个包含多个特征图的列表，
                              每个特征图对应一个不同的尺度（例如 13x13、26x26、52x52）。
                              每个特征图包含了每个网格单元的预测结果，包括边界框的坐标、置信度以及每个类别的概率
        :param image_shape: 图片的大小
        :param max_boxes: 最大物体框数，限制最终检测框的数量，通常用于限制输出检测框的数量
        :return: box_: 物体框的位置
                 scores: 物体类别的概率
                 classes_: 物体类别
        '''

        # 每个特征图使用的锚框索引
        # 第一层特征图使用 [6, 7, 8] 索引的锚框
        # 第二层特征图使用 [3, 4, 5] 索引的锚框
        # 第三层特征图使用 [0, 1, 2] 索引的锚框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        boxes = []  # 检测到的物体框
        box_scores = []  # 物体框对应的得分

        # input_shape 是 416x416
        # image_shape 是实际图片的大小

        # yolo_outputs[0] 对应第一个特征图 即 (batch_size,13,13,255)
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32  # (416,416)

        # 对三个特征层的输出获取每个预测 box 坐标和 box 分数，score = 置信度 x 类别概率
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        #   计算 3 个特征层对应框在原图的位置以及类别置信度得分
        # ---------------------------------------#
        for i in range(len(yolo_outputs)):  # len(yolo_outputs) = 3
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)

        # 放在一行里面便于操作
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold  # 布尔张量 存储 True False
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)

        # 存储最终经过非极大值抑制后的检测框、分数和类别
        boxes_ = []
        scores_ = []
        classes_ = []

        # ---------------------------------------#
        #   1、取出每一类得分大于 self.obj_threshold 的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        for c in range(len(self.class_names)):
            # 取出所有类为 c 的box
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            # 取出所有类为 c 的分数
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

            # 非极大值抑制
            '''
            NMS 算法的步骤：
            1. 计算每个框的得分。
            2. 按照得分从高到低排序。
            3. 选择得分最高的框，然后计算它与其他框的 IOU（重叠度）。
            4. 将与当前框的 IOU 大于 iou_threshold 的其他框抑制。
            5. 重复上述步骤，直到所有框都处理完
            '''
            # 返回值 nms_index 是保留下来的框对应的索引
            nms_index = tf.image.non_max_suppression(class_boxes,  # 物体框坐标 (N, 4) 张量
                                                     class_box_scores,  # 物体框的得分 (N,) 张量
                                                     max_boxes_tensor,  # 最大保留的框数量
                                                     iou_threshold=self.nms_threshold  # IOU 阈值，决定重叠多少就认为是重复框
                                                     )
            # 获取非极大值抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            # 创建一个与 class_box_scores 相同形状的张量，将所有框的类别设为当前类别 c
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    # ---------------------------------------#
    #   predict用于预测，分三步
    #   1、建立yolo对象
    #   2、获得预测结果
    #   3、对预测结果进行处理
    # ---------------------------------------#
    def predict(self, inputs, image_shape):
        '''
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
        :param inputs: 处理之后的图片大小
        :param image_shape: 图像原始大小
        :return: boxes: 物体框坐标
                 scores: 物体概率值
                 classes: 物体类别
        '''
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)

        # yolo_inference 用于获得网络的预测结果
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes
