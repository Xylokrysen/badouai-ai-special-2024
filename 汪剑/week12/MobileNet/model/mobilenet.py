# -----------------------------------------------#
#        MobileNet的网络部分
# -----------------------------------------------#
import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def _conv_block(inputs,
                filters,
                kernel=(3, 3),
                strides=(1, 1)):
    x = Conv2D(filters, kernel, strides=strides, padding='same', use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)  # 使用 relu6，用于 MobileNet 等轻量级网络，避免数值过大，提高计算稳定性
    return x


def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
    '''
    Depthwise 卷积（3×3）：单独对每个通道进行卷积，减少计算量
    DepthwiseConv2D 仅对每个通道独立卷积，不会混合通道信息，它的输出仍然是多个通道的特征图

    depth_multiplier=1：输入通道数 = 输出通道数。
    depth_multiplier>1：输入通道数 × depth_multiplier = 输出通道数
    '''
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,  # 控制每个输入通道的卷积核个数，默认为 1（即每个通道仅有 1 个卷积核）
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    '''
    Pointwise 卷积（1×1）：整合通道信息，改变通道数  
    '''
    x = Conv2D(filters=pointwise_conv_filters, kernel_size=(1, 1),
               padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

    return x


def relu6(x):
    return K.relu(x, max_value=6)


def MobileNet(input_shape=[224, 224, 3],
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):
    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)

    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)

    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)  # 输出形状 (1024,)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)  # 使用 1×1 Pointwise 卷积代替全连接层。Conv2D(1×1) 作用类似于 Dense()，但计算效率更高
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs,x,name='mobilenet_1_0_224_tf')

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet(input_shape=(224,224,3))

    model.load_weights('mobilenet_1_0_224_tf.h5')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    print('Input image shape: ',x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted: ',decode_predictions(preds,1))
