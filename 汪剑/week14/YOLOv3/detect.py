import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageDraw, ImageFont
from YOLOv3.utils import load_weights, letterbox_image

# 指定使用 GPU 的 index
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index


def detect(image_path, model_path, yolo_weights=None):
    '''
    Introduction
    ------------
        加载模型，进行预测
    Parameters
    ----------
    :param image_path: 图片路径
    :param model_path: 模型路径，当使用 yolo_weights 无用
    :param yolo_weights: 权重
    :return:
    '''
    # ---------------------------------------#
    #   图片预处理
    # ---------------------------------------#
    image = Image.open(image_path)  # 对应size显示为 (w,h) 而OpenCV 对应shape显示为 (h,w,c)
    # 对预测的图像按照长度比进行缩放，不足的地方进行填充
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, np.float32)
    # 归一化
    image_data /= 255.
    # 转格式，第一维度填充
    image_data = np.expand_dims(image_data, axis=0)  # 形状变成 (1,h,w,c)

    # ---------------------------------------#
    #   图片输入
    # ---------------------------------------#
    # input_image_shape 原图的 size
    input_image_shape = tf.placeholder(dtype=tf.float32, shape=(2,))
    # 图像
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)

    # 进入 yolo_predictor 进行预测，yolo_predictor 是用于预测的一个对象
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)

    with tf.Session() as sess:
        # ---------------------------------------#
        #   图片预测
        # ---------------------------------------#
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)

            # 载入模型
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)

            # 进行预测
            out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
                                                          feed_dict={
                                                              # image_data这个resize过
                                                              input_image: image_data,
                                                              # 以y、x的方式传入
                                                              input_image_shape: [image.size[1], image.size[0]]})
        else:
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            '''
            训练阶段保存模型：
            with tf.Session() as sess:
                # 假设模型已经训练好，训练的变量保存在模型中
                saver = tf.train.Saver()
                # 保存模型权重
                saver.save(sess, 'model.ckpt')
                
            推理阶段加载模型：
            with tf.Session() as sess:
                saver = tf.train.Saver()
                # 加载训练好的模型
                saver.restore(sess, 'model.ckpt')
                print("Model restored.")
                # 现在我们可以使用模型进行预测或进一步的操作
                
            总结：
            tf.train.Saver() 是用于加载 完整的模型检查点文件，包括所有变量（权重、偏置、优化器状态等）。
            适用于需要恢复训练状态或继续训练的场景
            加载预训练权重（如 yolo_weights）时，直接加载权重文件并将其赋值给模型对应的变量，
            这时不需要使用 Saver，因为我们不需要恢复优化器状态或训练状态，只是恢复模型的 权重部分
            '''
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={input_image: image_data,
                           input_image_shape: [image.size[1], image.size[0]]}
            )

        # ---------------------------------------#
        #   画框
        # ---------------------------------------#
        # 找到几个 box，打印
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体和大小
        font = ImageFont.truetype(font='./font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 设置框的线条厚度
        thickness = (image.size[0] + image.size[1]) // 300

        # 遍历每个框并绘制框和标签
        # enumerate(out_classes): 返回每个框的索引和对应框的类别索引
        # reversed: 倒序遍历，这样可以首先绘制得分较高的框（得分较高的框可能会覆盖低得分框，倒序可以确保框不会被覆盖）
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = predictor.class_names[c]  # 获取对应的类别名称
            box = out_boxes[i]  # 获取对应的边框位置 (ymin,xmin,ymax,xmax)
            score = out_scores[i]  # 获取当前框的置信度得分

            # 准备框和标签
            label = '{} {:.2f}'.format(predicted_class, score)

            # 用于画框框和文字
            draw = ImageDraw.Draw(image)
            # textsize 用于获得写字的时候，按照这个字体，需要多大的框。文本对应的宽度和高度
            label_size = draw.textsize(label, font)

            # 获得四个边
            top, left, bottom, right = box  # 分别对应：ymin,xmin,ymax,xmax
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1] - 1, np.floor(bottom + 0.5)).astype('int32')
            right = min(image.size[0] - 1, np.floor(right + 0.5)).astype('int32')

            print(label, (left, top), (right, bottom))
            print(label_size)

            # 计算文本的起始位置
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 绘制边界框和文本
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=predictor.colors[c]
                )

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=predictor.colors[c]
            )
            draw.text(text_origin, label, font=font)

            del draw  # 删除绘图对象，释放内存
        image.show()
        image.save('./img/result1.jpg')

if __name__ == '__main__':
    # 当使用 yolo3 自带的 weights 的时候
    if config.pre_train_yolo3 == True:
        detect(config.image_file,config.model_dir,config.yolo3_weights_path)

    # 当使用自训练模型的时候
    else:
        detect(config.image_file,config.model_dir)
