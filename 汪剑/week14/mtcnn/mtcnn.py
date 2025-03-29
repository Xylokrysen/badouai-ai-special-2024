from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Activation, Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
from week14.mtcnn import utils
import cv2


# -----------------------------#
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
# -----------------------------#
'''
PReLU 是 Parametric ReLU（带参数的 ReLU）激活函数，与标准 ReLU 相比，
它在输入为负值时引入了一个可学习的参数来控制斜率，从而让网络在训练过程中可以自动调整负区间的响应
shared_axes=[1,2]:
这表示该层在第1轴和第2轴（通常对应于图像的高度和宽度）上的参数（即负斜率α）是共享的，
也就是说，每个特征图（channel）使用同一个α，而不是为每个像素都独立学习一个参数
'''


def create_Pnet():
    input = Input(shape=[None, None, 3])

    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数线性
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])
    return model


# -----------------------------#
#   mtcnn的第二段
#   精修框
# -----------------------------#
def create_Rnet():
    input = Input(shape=[24, 24, 3])

    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='valid')(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([input], [classifier, bbox_regress])
    return model


# -----------------------------#
#   mtcnn的第三段
#   精修框并获得五个点
# -----------------------------#
def create_Onet():
    input = Input(shape=[48, 48, 3])

    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    # 10,10,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 3,3,128 -> 128,3,3
    x = Permute((3, 2, 1))(x)
    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    bbox_regress = Dense(4, name='conv6-2')(x)
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    return model


class mtcnn():
    def __init__(self, weight_path1, weight_path2, weight_path3):
        self.Pnet = create_Pnet()
        self.Rnet = create_Rnet()
        self.Onet = create_Onet()

        '''
        by_name=True:
        表示按照层的名称来加载权重。只有当前模型中与权重文件中具有相同名称的层会加载相应的权重，
        而不要求层的顺序或结构完全一致。这对于迁移学习或部分加载预训练模型权重非常有用，因为你可以只加载那些你关心的层的权重
        '''
        self.Pnet.load_weights(weight_path1, by_name=True)
        self.Rnet.load_weights(weight_path2, by_name=True)
        self.Onet.load_weights(weight_path3, by_name=True)

    def detectFace(self, img, threshold):
        # -----------------------------#
        #   归一化，加快收敛速度
        #   把[0,255]映射到(-1,1)
        # -----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape
        # -----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        # -----------------------------#
        scales = utils.calculateScales(img)

        out = []
        # -----------------------------#
        #   粗略计算人脸框
        #   pnet部分
        # -----------------------------#
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))  # 形状 (h,w,c)
            '''
            *scale_img.shape：使用 * 运算符将 shape 元组中的每个维度展开，相当于传入 reshape 函数的多个参数
            '''
            inputs = scale_img.reshape(1, *scale_img.shape)  # 添加一个批次维度，形状 (1,h,w,c)
            # 图像金字塔中的每张照片分别传入Pnet得到output
            output = self.Pnet.predict(inputs)
            # 将所有 output 加入 out
            out.append(output)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # 有人脸的概率
            '''
            out[i]: 取出第 i 个尺度的 P-Net 预测输出（包含分类和回归）
            out[i][0]: 取出分类分支 cls_prob，形状 (1, H, W, 2)
            out[i][0][0]: 去掉 batch 维度，变为 (H, W, 2)
            out[i][0][0][:,:,-1]: 取 cls_prob 的最后一个通道（即 [..., 1]），得到人脸的概率图，形状变为 (H, W)
            '''
            cls_prob = out[i][0][0][:, :, -1]
            # 其对应框的位置
            roi = out[i][1][0]  # (H, W, 4)

            # 取出每个缩放后图片的长度
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            print(cls_prob.shape)

            # 解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            # extend() 方法会将 rectangle 中的所有元素逐个添加到 rectangles，而不是作为一个整体添加
            rectangles.extend(rectangle)  # 等同于 rectangles += rectangle

        # 进行非极大值抑制
        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        # -----------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            # crop_img = copy_img[y1:y2,x1:x2]
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]): int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(24,24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   计算人脸框
        #   onet部分
        # -----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(48,48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])

        return rectangles
