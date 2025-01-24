import os
import tensorflow as tf

'''
Cifar-10数据集：
针对二进制版本 cifar-10-batches-bin
每个文件包含 10000 个图像的数据
每一行 3073 个字节数据：
第 1 个字节是每一张图像对应的标签值，范围：0-9
2-3073位置的 3072 个字节是图像像素值，按 RGB 形式存储：1024+1024+1024 字节                 
'''

num_classes = 10

# 设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的 Cifar-10 的数据
class CIFAR10Record(object):
    pass


# 定义一个读取 Cifar-10 的函数 read_cifar10() ，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1  # Cifar-10表示这个数据集包含 10 个类别的数据，此处为 1 ，如果 Cifar-100 ，则此处是 2
    result.height = 32
    result.width = 32
    result.depth = 3  # RGB三通道，所以深度是 3

    image_bytes = result.height * result.width * result.depth  # 图片样本单张图像的像素数量（32 × 32 × 3 = 3072）
    record_bytes = label_bytes + image_bytes  # 单条记录的总字节数 = 标签字节数（1）+ 图像字节数（3072）= 3073

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)  # 创建一个固定长度记录的读取器，每次读取 record_bytes 字节的数据
    result.key, value = reader.read(file_queue)  # 从文件队列中逐条读取记录,返回 key（文件名）和 value（原始二进制数据）

    record_bytes = tf.decode_raw(value, tf.uint8)  # 将二进制数据解析为无符号整型数组（uint8），解码后得到的是一个一维的 tf.Tensor，形状为 [N]

    # tf.strided_slice: 提取数组的前 label_bytes 个元素，即图像的分类标签
    # tf.cast: 将提取的标签转换为 int32 类型，方便后续处理
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # tf.reshape: 将提取的 1D 数据重塑为 3D 数组 [depth, height, width]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [1], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])

    # 将数组转换为 [height, width, depth] 格式（符合 TensorFlow 图像标准） CIFAR-10 存储格式为 [c,h,w]，而大多数处理逻辑需要 [h,w,c]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result  # 回值是已经把目标文件里面的信息都读取出来


# 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def inputs(data_dir, batch_size, distorted):
    # 拼接数据目录和文件名（data_batch_1.bin 到 data_batch_5.bin）
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1,6)]

    file_queue = tf.train.string_input_producer(filenames)  # 使用 tf.train.string_input_producer 创建一个输入队列
    read_input = read_cifar10(file_queue)  # 调用 read_cifar10 解析文件队列中的图像和标签

    # 将图像数据从 uint8 转换为 float32 以支持浮点计算
    # 大多数 TensorFlow 操作需要浮点数据进行运算
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    num_examples_per_epoch = num_examples_pre_epoch_for_train  # 定义每个 epoch 的样本数量


    if distorted != None:  # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理

        '''
        tf.random_crop 将图像裁剪为 [24, 24, 3]
        增强数据的多样性，模拟不同视角和尺寸的拍摄效果，防止过拟合
        '''
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])

        '''
        tf.image.random_flip_left_right 随机水平翻转图像
        模拟对称性（如动物、汽车等目标），提高模型对翻转样本的鲁棒性
        '''
        flipped_image = tf.image.random_flip_left_right(cropped_image)

        '''
        tf.image.random_brightness() 进行随机亮度调整
        模拟不同光照条件，增强模型在真实场景下的适应性
        '''
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)

        '''
        tf.image.random_contrast() 进行随机对比度调整
        模拟摄像头或环境条件下的对比度变化
        '''
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)

        # 进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        # 将数据调整为零均值和单位方差
        # 将像素值归一化，使其在 0 附近分布，适应模型输入要求，提高收敛速度
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        '''
        设置图片数据及标签的静态形状，保证后续批处理和计算图兼容性

        尽管 tf.random_crop 已经生成了大小为 [24, 24, 3] 的张量，但 TensorFlow 的某些操作不会自动推断静态形状，
        或者推断结果中包含未定义的维度（None）。
        为了让模型的计算图在构图阶段更清晰，手动调用 set_shape 显式定义张量形状是一个常见的做法
        '''
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        # 定义用于填充队列的最小样本数量
        # 确保训练队列中始终有足够的样本，避免因数据不足导致训练中断
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)

        # tf.train.shuffle_batch 从队列中随机取样，生成图像和标签批次
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size=batch_size,  # 每个批次的样本数量
                                                            num_threads=16,  # 使用的线程数量
                                                            capacity=min_queue_examples + 3 * batch_size,  # 队列容量
                                                            min_after_dequeue=min_queue_examples)  # 从队列中取出元素后保留的最小样本数，用于打乱

        # 使用 tf.train.shuffle_batch 后 label 形状变为 [batch_size,1] 因此需要调整为 [batch_size] 一维数组
        return images_train, tf.reshape(labels_train, [batch_size])

    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24,
                                                               24)  # tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切

        float_image = tf.image.per_image_standardization(resized_image)

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size,  # 每个批次的样本数量
                                                  num_threads=16,  # 使用的线程数量
                                                  capacity=min_queue_examples + 3 * batch_size)  # 队列容量

        return images_test, tf.reshape(labels_test, [batch_size])
