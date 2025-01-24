import tensorflow as tf

'''
VGG16的结构：
1. 一张原始图片被 resize 到 (224,224,3)
2. conv1 两次 [3,3] 卷积网络, 输出的特征层为 64 ,输出 net 为 (224,224,64) , 再 2X2 最大池化, 输出 net 为 (112,112,64)
3. conv2 两次 [3,3] 卷积网络, 输出的特征层为 128 ,输出 net 为 (112,112,128) , 再 2X2 最大池化, 输出 net 为 (56,56,128)
4. conv3 三次 [3,3] 卷积网络, 输出的特征层为 256 ,输出 net 为 (56,56,256) , 再 2X2 最大池化, 输出 net 为 (28,28,256)
5. conv3 三次 [3,3] 卷积网络, 输出的特征层为 512 ,输出 net 为 (28,28,512) , 再 2X2 最大池化, 输出 net 为 (14,14,512)
6. conv3 三次 [3,3] 卷积网络, 输出的特征层为 512 ,输出 net 为 (14,14,512) , 再 2X2 最大池化, 输出 net 为 (7,7,512)
7. 利用卷积的方式模拟全连接层, 效果等同, 输出 net 为 (1,1,4096). 共进行两次
8. 利用卷积的方式模式全连接层, 效果等同, 输出 net 为 (1,1,1000). 最后输出的就是每个类的预测 
'''

# ----------------------------------#
# VGG16网络部分
# ----------------------------------#


'''
tf.contrib.slim 和 tf.nn 对比：
tf.contrib.slim：高级封装，专注于简化代码和模型搭建。用于快速搭建网络模型，灵活性低。属于 contrib，实验性质，在 TensorFlow 2.x 已被移除
tf.nn：低级 API，提供细粒度控制。用于自定义网络模型，灵活性较高。核心模块，在 TensorFlow 2.x 仍可使用
'''

# 创建 slim 对象
slim = tf.contrib.slim


def vgg16(inputs, # 输入张量，形状为 (n,h,w,c)
          num_classes=1000, # 用于分类的类别数量
          is_training=True,  # 指定当前模式是训练 (True) 还是推理 (False) 模式
          dropout_keep_prob=0.5, # 在 dropout 操作中保留激活值的概率，默认为 0.5
          spatial_squeeze=True, # 是否在最后的输出中对空间维度进行 squeeze（通常在分类问题中设置为 True）。全卷积结果拍扁用于分类
          scope='vgg_16'   #变量的命名空间
          ):

    # 定义一个变量作用域，所有在此作用域中创建的变量都会带有前缀 'vgg_16'。[inputs] 定义作用域的默认输入张量
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # 建立 vgg_16 网络

        '''
        slim.repeat(inputs, repetitions, layer, *args, **kwargs)  重复指定的操作
        参数说明：
        inputs：输入的张量，作为 layer 操作的输入
        repetitions：指定重复的次数，即 layer 操作的调用次数
        layer：要重复的操作（通常是一个函数，比如 slim.conv2d 或 slim.max_pool2d）
        *args 和 **kwargs：传递给 layer 的其他参数，例如 num_outputs、kernel_size、stride 等
        
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        等价于
        net = slim.conv2d(inputs, 64, [3, 3], scope='conv1/conv2d_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv1/conv2d_2')
        '''

        '''
        slim.conv2d 默认padding格式：SAME
        slim.max_pool2d 默认padding格式：VALID
        '''
        # conv1 两次 [3,3] 卷积网络, 输出的特征层为 64 ,输出 net 为 (224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2X2 最大池化, 输出 net 为 (112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2 两次 [3,3] 卷积网络, 输出的特征层为 128 ,输出 net 为 (112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2 最大池化, 输出 net 为 (56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3 三次 [3,3] 卷积网络, 输出的特征层为 256 ,输出 net 为 (56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2 最大池化, 输出 net 为 (28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv3 三次 [3,3] 卷积网络, 输出的特征层为 512 ,输出 net 为 (28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2 最大池化, 输出 net 为 (14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv3 三次 [3,3] 卷积网络, 输出的特征层为 512 ,输出 net 为 (14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2 最大池化, 输出 net 为 (7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出 net 为 (1,1,4096)
        net = slim.conv2d(net,4096,[7,7],padding='VALID',scope= 'fc6')
        net = slim.dropout(net,dropout_keep_prob,is_training=is_training,scope= 'dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出 net 为 (1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        # 利用卷积的方式模拟全连接层，效果等同，输出 net 为 (1,1,1000)
        # activation_fn= None 指定不使用激活函数（输出为线性激活）
        # normalizer_fn=None 不使用正则化方法
        net= slim.conv2d(net,num_classes,[1,1],activation_fn= None,normalizer_fn = None,scope= 'fc8')

        # 由于卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            # net: 输入张量，形状为 [batch_size, 1, 1, num_classes]
            # [1, 2]: 移除第 1 和第 2 维（空间维度）
            net=tf.squeeze(net,[1,2],name='fc8/squeezed')
        return net
