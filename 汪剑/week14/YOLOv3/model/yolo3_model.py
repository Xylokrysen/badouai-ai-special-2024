import numpy as np
import tensorflow as tf
import os


class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        '''
        Introduction
        --------------
            初始化函数
        Parameters
        --------------
        :param norm_epsilon: 在预测时计算 moving average 的衰减率
        :param norm_decay: 方差加上极小的数，防止除以 0 的情况
        :param anchors_path: yolo anchor 文件路径
        :param classes_path: 数据集类别对应文件
        :param pre_train: 是否使用预训练 darknet53 模型
        '''
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    # -----------------------------------#
    #  获取种类和先验框
    # -----------------------------------#
    def _get_class(self):
        '''
        Introduction
        ------------------
            获取类别名字
        Retruns
        ------------------
            class_names: coco数据集类别对应的名字
        '''
        # expanduser 可以处理路径中的 ~，并自动将其转换为当前操作系统中用户的主目录路径 /home/user
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        '''
        Introduction
        ------------
            获取anchors
        '''
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)  # -1 是一个占位符，用来自动计算行数，确保数组的元素总数不变

    # -------------------------------#
    #   用于生成层
    # _______________________________#
    # l2 正则化
    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        '''
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
        :param input_layer: 输入的四维 tensor
        :param name: batchnorm 层的名字
        :param training: 是否训练过程
        :param norm_decay: 在预测时计算 moving average 时的衰减率
        :param norm_epsilon: 方差加上极小的数，防止除以0的情况
        :return: bn_layer: batch normalization 处理之后的 feature map
        '''
        '''
        momentum: momentum 是用于计算移动平均的衰减率。它影响均值和方差的更新速度，norm_decay 是一个变量，控制衰减的速度。
                  较大的 momentum 值会使得模型更多地依赖历史数据，较小的值会使模型更依赖当前批次数据
        epsilon: epsilon 是在归一化计算中添加的小常数，用于防止分母为零。norm_epsilon 是一个控制精度的变量，通常设为一个非常小的数值，比如 1e-5
        center: center 表示是否在批归一化过程中加上偏置（即通过学习一个偏置项来对输出进行调整）。
                设置为 True 时，批归一化层会学习一个偏置（也就是归一化后再加上一个可学习的参数），默认是 True
        scale: scale 表示是否在批归一化过程中使用可学习的缩放因子。设置为 True 时，批归一化层会学习一个缩放因子（一个可学习的参数），默认是 True
        training: training 是一个布尔值（通常为 True 或 False），用于指示当前模型是否在训练模式下。
                  当为 True 时，批归一化使用当前批次的均值和方差来进行归一化；
                  当为 False 时，使用训练过程中累积的均值和方差进行归一化（即使用移动平均）
        '''
        bn_layer = tf.layers.batch_normalization(inputs=input_layer,
                                                 momentum=norm_decay,
                                                 epsilon=norm_epsilon,
                                                 center=True,
                                                 scale=True,
                                                 training=training,
                                                 name=name)

        '''
        激活函数：Leaky ReLU
        Leaky ReLU(x) = x if x > 0 else ax
        '''
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)  # alpha=0.1, 这是 Leaky ReLU 的负数部分的斜率

    #  这个用来进行卷积
    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        '''
        Introduction
        ---------------
            使用 tf.layers.conv2d 减少权重和偏置矩阵初始化的过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行 batch norm，最后使用 leaky Relu 激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为 416*416，卷积核大小为 3，若 strides 为 2 时，(416 - 3)/2 + 1，计算结果为 208，相当于做了池化处理
            因此需要对 strides 大于 1 的时候，先进行一个 padding 操作，采用四周都 padding 一维代替 'same' 方式
        Parameters
        ------------------
        :param inputs: 输入变量
        :param filters_num: 卷积核数量
        :param kernel_size: 卷积核大小
        :param name: 卷积层名字
        :param use_bias: 是否使用偏置项
        :param strides: 卷积步长
        :return: conv: 卷积之后的 feature map
        '''
        '''
        kernel_initializer: 用于初始化卷积核的方式。tf.glorot_uniform_initializer() 表示使用的是 Glorot（Xavier）均匀分布初始化器
                            它将卷积核的初始值从一个均匀分布中随机抽取，范围为 
                            [-sqrt(6 / (input_num + output_num)),sqrt(6 / (input_num + output_num))]，input_num和output_num
                            分别是输入神经元数量和输出神经元数量。该初始化方式有助于避免梯度消失或梯度爆炸问题，特别是在深度网络中
        kernel_regularizer: 是应用于卷积核的正则化方法，这里使用的是 L2 正则化。L2 正则化会将卷积核的权重参数的平方和加到损失函数中，
                            正则化项会惩罚卷积核中的大权重，将权重大的值调整得更小，以防止过拟合。
                            scale=5e-4 设置了正则化的强度，即权重的正则化项将乘以 5e-4 作为缩放因子
        '''
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=filters_num,
                                kernel_size=kernel_size,
                                strides=(strides, strides),
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                padding=('same' if strides == 1 else 'valid'),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
                                use_bias=use_bias,
                                name=name)
        return conv

    # 这个用来进行残差卷积
    # 残差卷积就是进行一次 3x3 的卷积，然后保存该卷积 layer
    # 再进行一次 1x1 的卷积和一次 3x3 的卷积，并把这个结果加上 layer 作为最后的结果
    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.99,
                        norm_epsilon=1e-3):
        '''
        Introduction
        ------------
            Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
        Parameters
        ----------
        :return:
        :param inputs: 输入变量
        :param filters_num: 卷积核数量
        :param blocks_num: block数量
        :param conv_index: 为了方便加载预训练权重，统一命名序号
        :param training: 是否训练过程
        :param norm_decay: 在预测时计算 moving average 时的衰减率
        :param norm_epsilon: 方差加上极小的数，防止除以 0 的情况
        :return: inputs: 经过残差网络处理后的结果
        '''
        # 在输入 feature map 的长宽维度进行 padding
        '''
        tf.pad: paddings 定义了在每个维度上的填充量，paddings 是一个 4x2 的列表，
                分别对应张量的 4 个维度 [batch_size, height, width, channels]
                height（第二个维度）：[1, 0]，表示在 height（高度）方向的前面填充 1 行，后面不填充
                width（第三个维度）：[1, 0]，表示在 width（宽度）方向的前面填充 1 列，后面不填充
        '''
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name='conv2d_' + str(conv_index))
        layer = self._batch_normalization_layer(layer, name='batch_normalization_' + str(conv_index), training=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1, strides=1,
                                       name='conv2d_' + str(conv_index))
            layer = self._batch_normalization_layer(layer, name='batch_normalization_' + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name='conv2d_' + str(conv_index))
            layer = self._batch_normalization_layer(layer, name='batch_normalization_' + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    # -------------------------------------#
    #   生成 _darknet53 和逆卷积层
    # -------------------------------------#
    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        '''
        Introduction
        ------------
            构建yolo3使用的 darknet53 网络结构
        Parameters
        ----------
        :param inputs: 模型输入变量
        :param conv_index: 卷积层序号，方便根据名字加载训练权重
        :param training: 是否训练过程
        :param norm_decay: 在预测时计算 moving average 时的衰减率
        :param norm_epsilon: 方差加上极小的数，防止除以 0 的情况
        :return: conv: 经过 52 层卷积之后的结果，输入图片为 416x416x3，则此时输出的结果 shape 为 13x13x1024
                 route1: 返回第 26 层卷积计算结果 52x52x256，供后续使用
                 route2: 返回第 43 层卷积计算结果 26x26x512，供后续使用
                 conv_index: 卷积层计数，方便在加载预训练模型时使用
        '''
        with tf.variable_scope('darknet53'):
            # 416,416,3 -> 416,416,32
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1,
                                      name='conv2d_' + str(conv_index))
            conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index),
                                                   training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1

            # 416,416,32 -> 208,208,64
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=64, blocks_num=1,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)

            # 208,208,64 -> 104,104,128
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)

            # 104,104,128 -> 52,52,256
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 52,52,256
            route1 = conv

            # 52,52,256 -> 26,26,512
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 26,26,512
            route2 = conv

            # 26,26,512 -> 13,13,1024
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=1024, blocks_num=4,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            return route1, route2, conv, conv_index

    # 输出两个结果
    # 第一个是进行 5 次卷积后，用于下一次逆卷积的，卷积过程是 1x1,3x3,1x1,3x3,1x1
    # 第二个是进行 5+2 次卷积，作为一个特征层的，卷积过程是 1x1,3x3,1x1,3x3,1x1,3x3,1x1,3x3,1x1
    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99,
                    norm_epsilon=1e-3):
        '''
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的 feature map 的block，这样来提高对小物体的检测率
        Parameters
        ----------
        :param inputs: 输入特征
        :param filter_num: 卷积核数量
        :param out_filters: 最后输出层的卷积核数量
        :param conv_index: 卷积层数序号，方便根据名字加载训练过程
        :param training: 是否训练过程
        :param norm_decay: 在预测时计算 moving average 时的衰减率
        :param norm_epsilon: 方差加上极小的数，防止除以 0 的情况
        :return: route: 返回最后一层卷积的前一层结果
                 conv: 返回最后一层卷积的结果
                 conv_index: conv层计数
        '''
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv

        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name='conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization_' + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1,
                                  name='conv2d_' + str(conv_index),use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    # 返回三个特征层的内容
    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        '''
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
        :param inputs: 模型输入的张量
        :param num_anchors: 每个 grid cell 负责检测的 anchor 数量
        :param num_classes: 类别数量
        :param training: 是否训练过程
        :return: conv1: 13x13x75
                 conv2: 26x26x75
                 conv3: 52x52x75
        '''
        conv_index = 1
        # route1 = 52,52,256    route2 = 26,26,512  route3 = 13,13,1024
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training=training,
                                                                 norm_decay=self.norm_decay,
                                                                 norm_epsilon=self.norm_epsilon)
        with tf.variable_scope('yolo'):
            '''
            获得第一个特征层
            conv2d_57 = 13,13,512  conv2d_59 = 13,13,num_anchors * (num_classes + 5)
            '''
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)

            '''
            获得第二个特征层
            '''
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1,
                                           name='conv2d_' + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name='batch_normalization_' + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1

            # 进行上采样 13,13,256 -> 26,26,256
            # upSample_0 = 26,26,256
            upSample_0 = tf.image.resize_nearest_neighbor(conv2d_60,
                                                          [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]],
                                                          name='upSample_0')
            # route0 = 26,26,768
            route0 = tf.concat([upSample_0, conv2d_43], axis=-1, name='route_0')
            # conv2d_65 = 26,26,256  conv2d_67 = 26,26,num_anchors * (num_classes + 5)
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)

            '''
            获得第三个特征层
            '''
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1,
                                           name='conv2d_' + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name='batch_normalization_' + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1

            # 进行上采样 26,26,128 -> 52,52,128
            # upSample_1 = 52,52,128
            upSample_1 = tf.image.resize_nearest_neighbor(conv2d_68,
                                                          [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]],
                                                          name='upSample_1')
            # route1 = 52,52,384
            route1 = tf.concat([upSample_1, conv2d_26], axis=-1, name='route_1')

            # conv2d_73 = 52,52,128 conv2d_75 = 52,52,num_anchors * (num_classes + 5)
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index=conv_index,
                                               training=training, norm_decay=self.norm_decay,
                                               norm_epsilon=self.norm_epsilon)
        return [conv2d_59, conv2d_67, conv2d_75]
