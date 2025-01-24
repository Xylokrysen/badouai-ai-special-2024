# 构造神经网络整体结构，并进行训练和测试（评估）过程
import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = 'Cifar_data/cifar-10-batches-bin'

###############################################################################################
'''
通常在训练过程中，总损失 = 原始损失(数据误差，提高训练数据拟合) + 正则化损失(权重约束，限制权重的幅度，避免过拟合，增强模型泛化能力)
'''
###############################################################################################

'''
创建一个 variable_with_weight_loss() 函数，该函数的作用是：
 1.使用参数 w1 控制 L2 loss 的大小
 2.使用函数 tf.nn.l2_loss() 计算权重 L2 loss
 3.使用函数 tf.multiply() 计算权重 L2 loss 与 w1 的乘积，并赋值给 weight_loss
 4.使用函数 tf.add_to_collection() 将最终的结果放在名为 losses 的集合里面，方便后面计算神经网络的总体loss
'''

'''
以下函数的作用：初始化权重，计算模型参数（权重）正则化损失
典型形式：
  1. L2 正则化（权重平方和的一半），也称为权重衰减
  2. L1 正则化（权重绝对值和）
'''


def variable_with_weight_loss(shape, stddev, w1):
    '''
    tf.truncated_normal 用于生成服从截断正态分布随机数的函数，它会生成正态分布的随机数，但会将那些超过两个标准差的值截断
    在 TensorFlow 2.x 中使用 tf.random.truncated_normal
    参数说明：
    shape: 指定生成张量的形状（例如 [2, 3]）。
    mean: 指定正态分布的均值，默认为 0.0。
    stddev: 指定正态分布的标准差，默认为 1.0。
    dtype: 指定生成张量的类型（例如 tf.float32）。
    seed: 随机种子，用于控制随机性。
    name: 操作的名称（可选）
    '''
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # tf.truncated_normal 初始化神经网络权重，避免极值问题
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')  # 计算正则化损失
        tf.add_to_collection('losses', weight_loss)  # 权重衰减损失 weight_loss 添加到名为 'losses' 的集合中
    return var


# 使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据文件
# 其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 创建 x 和 y_ 两个 placeholder ，用于在训练或评估时提供输入的数据和对应的标签值
# 要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])  # (n,h,w,c)
y_ = tf.placeholder(tf.int32, [batch_size])

# mini-batch training
# 创建第一个卷积层 shape = (kh,kw,ci,co)
kernel1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2,
                                    w1=0.0)  # 64个卷积核形状是 [5,5,3],相当于输出 64 通道，64 个特征图（feature map）
'''
tf.nn.conv2d(
    input,           # 输入数据, [batch_size, height, width, channels],默认格式为 NHWC
    filters,         # 卷积核（过滤器）
    strides,         # 滑动步幅,通常为 [1, stride_height, stride_width, 1],stride_height 和 stride_width 分别表示卷积核在高度和宽度方向上的移动步长
    padding,         # 填充方式, SAME: 卷积的输出尺寸与输入尺寸一致(步幅为1), VALID: out_size = (in_size−filter_size) / stride + 1
    use_cudnn_on_gpu=None,  # 是否在 GPU 上使用 CuDNN（默认自动）
    data_format=None,       # 数据格式（默认是 NHWC 格式）, NCHW 格式通常在 GPU 上更高效
    name=None               # 操作的名称（可选）
)
'''
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1],
                     padding='SAME')  # 输出的形状为 [batch_size, out_height, out_width, out_channels]
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))  # 输出形状 (batch_size,24,24,64)

'''
tf.nn.max_pool(
    input,       # 输入张量
    ksize,       # 池化窗口的大小, ksize 通常为 [1, 2, 2, 1] 或 [1, 3, 3, 1]
    strides,     # 池化窗口的滑动步长, strides 通常为 [1, 2, 2, 1]
    padding,     # 填充方式 ('SAME' 或 'VALID') 
                   SAME填充：out_height(out_width) = in_height(in_width) / stride_height  
                   VALID填充：out_height(out_width) = (in_height(in_width) - filter_height) / stride_height + 1
    name=None    # 操作名称（可选）
)
'''
# 最大池化用于降低特征图的空间尺寸（宽度和高度），从而减少计算量和过拟合，同时提取最显著的特征
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 输出形状 (batch_size,12,12,64)

# 创建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 输出形状 [batch_size,6,6,64]

# 因为要进行全连接层操作，所以这里使用 tf.reshape() 函数将 pool2 输出变成一维向量，并使用 get_shape() 函数获取扁平化之后的长度
reshape = tf.reshape(pool2, [batch_size, -1])  # 这里的-1代表将pool2的三维结构拉直为一维结构
dim = reshape.get_shape()[1].value  # get_shape()[1].value 表示获取 reshape 之后的第二个维度的值

# 建立第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 建立第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(local4,  ), fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失

'''
tf.nn.softmax_cross_entropy_with_logits 该损失函数结合了 softmax 激活和 交叉熵 损失的计算
'''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))

weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))

'''
tf.reduce_mean(cross_entropy)：计算交叉熵损失的平均值
tf.reduce_mean 会将其沿着 batch 维度求平均，得到整个批次的平均损失
'''
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 函数 tf.nn.in_top_k() 用来计算输出结果中 top_k 的准确率，函数默认的 k 值是 1 ，即 top 1 的准确率，也就是输出分布准确率最高时的数值
top_k_op = tf.nn.in_top_k(result, y_, 1)  # 返回的结果 [batch_size] 每个元素为 True 或者 False

# 变量初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作，这是因为之前数据增强的时候使用 train.shuffle_batch() 函数的时候通过参数 num_threads() 配置了 16 个线程用于组织 batch 的操作
    tf.train.start_queue_runners()

    # 每隔 100 step 会计算并展示当前的loss，每秒钟能训练的样本数量、以及训练一个 batch 数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time  # 计算当前批次的训练耗时

        if step % 100 == 0:
            examples_per_sec = batch_size / duration  # 每秒钟处理的样本数量，衡量训练速度
            sec_per_batch = float(duration)  # 训练一个批次的耗时，衡量每次前向传播和反向传播所需时间
            print('step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)' % (
            step, loss_value, examples_per_sec, sec_per_batch))

    # 计算最终的准确率
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # 向上取整
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch = sess.run([images_test,labels_test])
        predictions = sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print('accuracy = %.3f%%' % ((true_count/total_sample_count) * 100))
