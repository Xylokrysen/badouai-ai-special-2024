# InceptionV3的网络部分

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

'''
from __future__ import print_function: 
从 __future__ 模块导入功能，确保代码兼容 Python 2 和 Python 3。使用 Python 3 风格的 print() 函数，而不是 Python 2 的 print 语句

from __future__ import absolute_import: 
强制使用绝对导入，避免相对导入的混淆。在Python 2.x中，允许使用相对导入，但在Python 3.x中，相对导入的行为有所不同。通过from __future__ import absolute_import，你可以保证导入时总是使用绝对路径（即从顶级模块开始的路径）。

import warnings:
导入warnings模块，该模块允许控制警告的生成和显示。通常用于显示警告消息或抑制警告，特别是在实验或开发中可能会用到，比如某些函数的弃用警告。


Keras是一个用于构建和训练深度学习模型的高层API，它通常作为TensorFlow的高级接口来使用。
from keras.models import Model: 导入Keras中的Model类，用于定义和训练神经网络模型。

from keras import layers: 导入Keras的layers模块，用于创建不同的神经网络层。

from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D:
Activation: 应用激活函数（如ReLU、Sigmoid等）到层输出。
Dense: 全连接层，用于神经网络中的每个神经元连接。
Input: 用于定义模型输入的层。
BatchNormalization: 批量归一化层，用于加速训练和提高模型稳定性。
Conv2D: 二维卷积层，常用于图像处理。
MaxPooling2D: 二维最大池化层，用于减少空间维度。
AveragePooling2D: 二维平均池化层，类似于最大池化，但使用平均值进行池化。

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D:
GlobalAveragePooling2D: 全局平均池化层，将每个特征图的所有值取平均，以减少维度。
GlobalMaxPooling2D: 全局最大池化层，类似于全局平均池化，但取特征图的最大值。

from keras.engine.topology import get_source_inputs: 
导入get_source_inputs函数，用于从模型中提取输入层。这个函数常用于获取Keras模型的输入层，尤其在定义自定义层时，可能需要用到。

from keras.utils.layer_utils import convert_all_kernels_in_model: 
导入convert_all_kernels_in_model函数，用于将模型中的所有卷积层的内核（滤波器）转换为某种格式。该函数通常用于模型的硬件加速或优化，尤其是在迁移到不同平台时。

from keras.utils.data_utils import get_file: 
导入get_file函数，用于下载文件并缓存。通常用于下载外部数据集或模型文件，并且确保文件在本地缓存，以避免重复下载。

from keras import backend as K: 
导入Keras的后端接口，通常用于访问底层张量操作的API。

from keras.applications.imagenet_utils import decode_predictions: 
导入decode_predictions函数，用于将模型的预测结果（如ImageNet分类）解码为可读标签。该函数通常用于将神经网络分类模型的输出（通常是概率分布）转换为实际的标签名称。

from keras.preprocessing import image: 
导入Keras中的image模块，用于图像预处理。这个模块包含了加载和预处理图像的函数，如load_img、img_to_array等，常用于数据输入到深度学习模型前的处理。
'''

'''
InceptionV3 网络部分：
    类型           kernel尺寸/步长（或注释）        输入尺寸         输出尺寸
    卷积                 3*3/2                 299*299*3       149*149*32
    卷积                 3*3/1                 149*149*32      147*147*32
    卷积                 3*3/1                 147*147*32      147*147*64
    池化                 3*3/2                 147*147*64      73*73*64
    卷积                 1*1/1                 73*73*64        73*73*80
    卷积                 3*3/2                 73*73*80        71*71*192
    池化                 3*3/2                 71*71*192       35*35*192
Inception模块组      3个Inception Module        35*35*192       35*35*288
Inception模块组      3个Inception Module        35*35*288       17*17*768
Inception模块组      3个Inception Module        17*17*768       8*8*2048
    池化                  8*8                  8*8*2048        1*1*2048
    线性                 logits                1*1*2048        1*1*1000
   Softmax              分类输出                1*1*1000        1*1*1000
'''


# 将卷积、归一化、激活3个处理步骤合到一起方便处理
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              strides=(1, 1),
              padding='same',
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu')(x)
    return x


def InceptionV3(input_shape=(299, 299, 3), classes=1000):
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')  # 输出形状 (149,149,32)
    x = conv2d_bn(x, 32, 3, 3, padding='valid')  # 输出形状 (147,147,32)
    x = conv2d_bn(x, 64, 3, 3)  # 输出形状 (147,147,64)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 输出形状 (73,73,64)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')  # 输出形状 (73,73,80)
    x = conv2d_bn(x, 192, 3, 3, padding='valid')  # 输出形状 (71,71,192)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 输出形状 (35,35,192)

    '''
    Inception模块组 
    Block1 35x35
    '''
    # Block1 part1
    # (35,35,192) -> (35,35,256)
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 输出形状 (35,35,64)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)  # 输出形状 (35,35,64)

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 输出形状 (35,35,96)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)  # 输出形状 (35,35,32)

    # 64 + 64 + 96 + 32 = 256
    # axis=3 表示沿着 通道轴（channels axis） 拼接 -> (n, h, w, c)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db1, branch_pool], axis=3, name='mixed0')

    # Block1 part2
    # (35,35,256) -> (35,35,288)
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 输出形状 (35,35,64)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)  # 输出形状 (35,35,64)

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 输出形状 (35,35,96)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)  # 输出形状 (35,35,64)

    # 64 + 64 + 96 + 64 = 288
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db1, branch_pool], axis=3, name='mixed1')

    # Block1 part3
    # (35,35,288) -> (35,35,288)
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 输出形状 (35,35,64)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)  # 输出形状 (35,35,64)

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)  # 输出形状 (35,35,96)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)  # 输出形状 (35,35,64)

    # 64 + 64 + 96 + 64 = 288
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db1, branch_pool], axis=3, name='mixed2')

    '''
    Inception模块组 
    Block2 17x17
    '''
    # Block2 part1
    # (35,35,288) -> (17,17,768)
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')  # 输出形状 (17,17,384)

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3, strides=(2, 2), padding='valid')  # 输出形状 (17,17,96)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 输出形状 (17,17,288)

    # 384 + 96 + 288 = 768
    x = layers.concatenate([branch3x3, branch3x3db1, branch_pool], axis=3, name='mixed3')

    # Block2 part2
    # (17,17,768) -> (17,17,768)
    branch1x1 = conv2d_bn(x, 192, 1, 1)  # 输出形状 (17,17,192)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)  # 输出形状 (17,17,192)

    branch7x7db1 = conv2d_bn(x, 128, 1, 1)
    branch7x7db1 = conv2d_bn(branch7x7db1, 128, 7, 1)
    branch7x7db1 = conv2d_bn(branch7x7db1, 128, 1, 7)
    branch7x7db1 = conv2d_bn(branch7x7db1, 128, 7, 1)
    branch7x7db1 = conv2d_bn(branch7x7db1, 192, 1, 7)  # 输出形状 (17,17,192)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)  # 输出形状 (17,17,192)

    # 192 + 192 + 192 + 192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db1, branch_pool], axis=3, name='mixed4')

    # Block2 part3 and part4
    # (17,17,768) -> (17,17,768) -> (17,17,768)
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)  # 输出形状 (17,17,192)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)  # 输出形状 (17,17,192)

        branch7x7db1 = conv2d_bn(x, 160, 1, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 7, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 1, 7)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 7, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 192, 1, 7)  # 输出形状 (17,17,192)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)  # 输出形状 (17,17,192)

        # 192 + 192 + 192 + 192 = 768
        x = layers.concatenate([branch1x1, branch7x7, branch7x7db1, branch_pool], axis=3, name='mixed' + str(5 + i))

    # Block2 part5
    # (17,17,768) -> (17,17,768)
    branch1x1 = conv2d_bn(x, 192, 1, 1)  # 输出形状 (17,17,192)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)  # 输出形状 (17,17,192)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)  # 输出形状 (17,17,192)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)  # 输出形状 (17,17,192)

    # 192 + 192 + 192 + 192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    '''
    Inception模块组
    Block3 8x8
    '''
    # Block3 part1
    # (17,17,768) -> (8,8,1280)
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')  # 输出形状 (8,8,320)

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')  # 输出形状 (8,8,192)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)  # 输出形状 (8,8,768)

    # 320 + 192 + 768 =
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # Block3 part2 and part3
    # (8,8,1280) -> (8,8,2048) -> (8,8,2048)
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)  # 输出形状 (8,8,320)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))  # 输出形状 (8,8,768)

        branch3x3db1 = conv2d_bn(x, 448, 1, 1)
        branch3x3db1 = conv2d_bn(branch3x3db1, 384, 3, 3)
        branch3x3db1_1 = conv2d_bn(branch3x3db1, 384, 1, 3)
        branch3x3db1_2 = conv2d_bn(branch3x3db1, 384, 3, 1)
        branch3x3db1 = layers.concatenate([branch3x3db1_1, branch3x3db1_2], axis=3)  # 输出形状 (8,8,768)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)  # 输出形状 (8,8,192)

        # 320 + 768 + 768 + 192 = 2048
        x = layers.concatenate([branch1x1, branch3x3, branch3x3db1, branch_pool], axis=3, name='mixed' + str(9 + i))

    # 平均池化后全连接
    '''
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    相当于：
    x = AveragePooling2D((8,8),name='avg_pool')(x)
    x = Flatten()(x)
    '''
    x = GlobalAveragePooling2D(name='avg_pool')(x)  # 输出形状 (2048,)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    model = Model(inputs, x, name='inception_v3')

    return model


def preprocess_input(x):
    x /= 255. # 将像素值从 [0, 255] 归一化到 [0, 1]（标准化到 0-1 之间）
    x -= 0.5  # 将范围平移到 [-0.5, 0.5]（使数据中心对齐到 0）
    x *= 2.   # 将数据缩放到 [-1, 1]，让数据更加适配 Inception 相关模型的输入格式。
    return x


if __name__ == '__main__':
    model = InceptionV3()

    model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))  # 返回的是一个 PIL 图像对象：<class 'PIL.Image.Image'>
    x = image.img_to_array(img)  # 将PIL对象转换成数组格式 (229,229,3)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted: ', decode_predictions(preds))
