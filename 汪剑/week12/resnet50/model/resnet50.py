# ResNet50网络部分


from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

'''
from __future__ import print_function: 
从 __future__ 模块导入功能，确保代码兼容 Python 2 和 Python 3。使用 Python 3 风格的 print() 函数，而不是 Python 2 的 print 语句

from keras import layers: 提供各种神经网络层，如卷积层、全连接层等。

from keras.layers import ...:
Input: 定义模型的输入张量（通常在函数式 API 模型中使用）。
Dense: 全连接层，用于普通的神经网络层。
Conv2D: 2D 卷积层，用于处理图像数据，进行特征提取。
MaxPooling2D: 最大池化层，用于降维和提取特征。
ZeroPadding2D: 零填充层，用于在图像边界增加填充值（例如用于保持特征图的大小）。
AveragePooling2D: 平均池化层，作用类似于最大池化，但使用平均值代替最大值。
Activation: 激活函数层，如 ReLU、sigmoid 等。
BatchNormalization: 批量归一化层，加速训练和提高稳定性。
Flatten: 将多维张量展开为一维张量（常用于全连接层前）。

from keras.models import Model: 用于构建复杂的神经网络结构。

from keras.preprocessing import image: 
image: 提供图像数据预处理工具。用途: 加载图像文件并转换为适合模型输入的格式，数据增强（旋转、缩放、平移等操作）。

import keras.backend as K:
keras.backend: Keras 的后端接口，用于调用底层深度学习框架（如 TensorFlow、Theano 或 CNTK）的操作
用途: 直接操作张量，获取后端相关信息（如默认张量类型、计算设备等）。

from keras.utils.data_utils import get_file:
get_file: 从指定的 URL 下载文件，并将其缓存到本地。用途: 通常用于下载预训练模型的权重文件。

from keras.applications.imagenet_utils import decode_predictions:
decode_predictions: 用于将模型输出的概率（通常是 logits）转换为对应的类标签。用途: 主要用于 ImageNet 分类模型的结果解码。

from keras.applications.imagenet_utils import preprocess_input:
preprocess_input: 用于预处理输入数据，使其符合预训练模型的输入要求（如归一化或去均值处理）。
用途: 将图像数据标准化（如减去 ImageNet 数据集的均值），确保输入数据的格式与模型一致。
'''

'''
Identity Block 输入和输出的维度相等，可以串联，用于加深网络
'''

'''
ResNet50 网络结构：
1. input ——> Zeropad ——> Conv2d ——> BatchNormalization ——> ReLU ——> MaxPool 输入(224,224,3) 输出 (56,56,64)
2. Conv Block ——> 2 个 Identity Block 串联
3. Conv Block ——> 3 个 Identity Block 串联
4. Conv Block ——> 5 个 Identity Block 串联
5. Conv Block ——> 2 个 Identity Block 串联
6. AveragePooling2D ——> Flatten ——> FC ——> Output
'''


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 左边支线（主分支）
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 右边支线（恒等映射），无任何处理。输入 input_tensor 即为输出 input_tensor

    # 两支线 tensor 相加
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


'''
Conv Block 输入和输出的维度是不一样的，所以不能串联，它的作用是改变网络维度
'''


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 左边支线（主干线）
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 右边支线
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    # 相加
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def ResNet50(input_shape=(224, 224, 3), classes=1000):
    # 定义模型输入的张量
    img_input = Input(shape=input_shape)  # Input 层 将形状信息传递给计算图，因此必须先定义张量，才能将数据传递给下一层处理

    '''
    填充的作用：通常，零填充是为了保持图像在经过卷积操作后不会太快缩小（减少特征图的尺寸），或者是为了让卷积操作能适应图像边界的情况。

    ResNet50 网络中使用零填充是为了确保网络的结构能够稳定地处理图像的边缘，避免卷积操作丢失边界信息
    '''
    x = ZeroPadding2D((3, 3))(img_input)  # 输出的形状为 (230,230,3)

    # 最开始的处理
    # Conv2D 默认 padding 模式：VALID。这里计算方式：floor((230 - 7) / 2 + 1) = 112
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)  # 输出的形状为 (112,112,64)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    # MaxPooling2D 默认 padding 模式：VALID。这里计算方式：floor((112 - 3) / 2 + 1) = 55
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 输出的形状为 (55,55,64)

    # kernel_size = (3,3) 和 kernel_size = 3 等价
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))  # 输出形状为 (55,55,256)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')  # 输出形状为 (55,55,256)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')  # 输出形状为 (55,55,256)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')  # 输出形状为 (28,28,512)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')  # 输出形状为 (28,28,512)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')  # 输出形状为 (14,14,1024)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')  # 输出形状为 (14,14,1024)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')  # 输出形状为 (7,7,2048)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')  # 输出形状为 (7,7,2048)

    '''
    针对 MaxPooling2D 和 AveragePooling2D 如果不指定 strides ，则默认 pool_size = strides。pool_size 默认值 (2,2)
    '''
    x = AveragePooling2D((7, 7), name='avg_pool')(x)  # 输出形状为 (1,1,2048)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    return model


# predict
if __name__ == '__main__':
    model = ResNet50()
    model.load_weights('./resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    model.summary()  # 打印出模型的概要信息，包括每一层的名称、类型、输出形状和参数数量等。这有助于了解模型结构

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 调用 preprocess_input 函数对图像数据进行预处理

    print('Input image shape: ', x.shape)
    preds = model.predict(x)  # 输出长度1000的概率分布数组
    # print(preds)
    print('Predicted: ', decode_predictions(preds))

    # 打印结果：[(class_id, class_name, class_probability),...]
    # [[('n02504458', 'African_elephant', 0.76734334),
    # ('n01871265', 'tusker', 0.1993866),
    # ('n02504013', 'Indian_elephant', 0.03216044),
    # ('n02410509', 'bison', 0.0005231227),
    # ('n02408429', 'water_buffalo', 0.00030516039)]]


