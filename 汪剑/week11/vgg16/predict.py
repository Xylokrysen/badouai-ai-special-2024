import vgg.vgg16
import tensorflow as tf
import numpy as np
import utils


# 读取图片
img = utils.load_image('./test_data/dog.jpg')

# 对输入的图片进行 resize，使其 shape 满足 (-1,224,224,3)
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(img, (224, 224))

# 建立网络结构
prediction = vgg.vgg16(resized_img)

# 载入模型
sess = tf.Session()

ckpt_filename= './model/vgg_16.ckpt'

sess.run(tf.global_variables_initializer())

# 加载预训练模型
saver =tf.train.Saver() # 用于保存和恢复模型
saver.restore(sess,ckpt_filename) # restore 将模型权重从文件 ckpt_filename 恢复到当前计算图

# 最后结果进行预测
'''
pro= tf.nn.softmax(prediction)
假设 prediction 的输出是：[2.0, 1.0, 0.1]。但是这一步只是搭建计算图，并不会实际计算。
直到 sess.run 被调用，才会最后实际输出结果
'''
pro= tf.nn.softmax(prediction)

pre=sess.run(pro,feed_dict={inputs:img})

# 打印预测结果
print('result: ')
utils.print_answer(pre[0],'./synset.txt')

sess.close()
