import cv2

'''
TensorBoard: 用于将训练过程中产生的指标（如损失、准确率等）可视化到 TensorBoard

ModelCheckpoint: 用于在训练过程中保存模型的权重。可以设置条件（如最低验证损失）来保存最佳模型
参数：
filepath：保存路径。
monitor：监控的指标（如 val_loss）。
save_best_only：是否仅保存最佳模型

ReduceLROnPlateau: 当监控指标停止改善时，动态降低学习率以提高模型收敛能力
参数：
monitor：监控的指标（如 val_loss）。
factor：每次降低学习率的倍数。
patience：指标未改善的轮数后减少学习率

EarlyStopping: 当监控的指标在若干轮内未改善时，提前停止训练以节省时间
参数：
monitor：监控的指标（如 val_loss）。
patience：指标未改善的轮数。
restore_best_weights：是否恢复到最佳权重
'''
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils  # 提供实用工具函数，常用于数据预处理和转换
from keras.optimizers import Adam
from model.Alexnet import Alexnet
import utils
import numpy as np

'''
导入 Keras 的后端模块，提供与底层深度学习框架（如 TensorFlow）的交互接口
常见用途：
定义自定义损失函数或激活函数。
获取张量的形状或数值。
切换不同后端（如 TensorFlow、Theano）
'''
from keras import backend as K

# K.set_image_dim_ordering('tf')
'''
在深度学习中，图像数据有两种格式：
channels_first: (batch, channels, height, width)
channels_last:  (batch, height, width, channels) ---- keras默认格式
'''
K.image_data_format() == 'channels_first'  # 检查图像数据格式是否是 'channels_first'，返回 True 或者 False

'''
定义一个数据生成器函数，用于批量读取和处理数据
lines: 包含数据路径或标签信息的列表，每个元素是一个字符串。格式为: image_name;label
batch_size: 指定每次生成的数据批量大小
返回值: 一个迭代器，生成包含图像和标签的批量数据
'''


def generate_arrays_from_file(lines, batch_size):
    n = len(lines)  # 获取总长度（即行数）
    i = 0  # 当前行的索引，用于迭代数据。初始化计数器

    # while True:
    while 1:  # 创建一个无限循环的生成器。数据生成器通常与模型训练结合使用（如 model.fit_generator()），需要不断提供批量数据，直到训练结束
        X_train = []  # 存储输入数据（图像）
        Y_train = []  # 存储标签数据

        # 获取一个 batch_size 大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)  # 当索引为 0 时，对 lines 中的数据进行随机打乱。确保每次训练或迭代中数据顺序不同，提高模型的泛化能力
            name = lines[i].split(';')[0]  # 从 lines[i] 提取图像文件名，文件格式：image_name;label
            # 从文件中读取图像
            img = cv2.imread(r'.\data\image\train' + '/' + name)  # 使用 OpenCV 读取图像，路径为 .data/image/train/<name>
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 默认使用 BGR
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])

            # 读完一个周期后重新开始
            i = (i + 1) % n

        # 处理图像
        X_train = utils.resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)  # -1: 自动计算批次大小（batch_size）

        # 将标签数据 Y_train 转换为 One-Hot 编码。num_classes=2 表示最终分为两类
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)

        # 返回一个批次的数据
        yield (X_train, Y_train)  # X_train 形状 (batch_size,224,224,3)  Y_train 形状 (batch_size,2)


'''
return 和 yield 的区别：
return 功能：return 立即结束函数的执行，并返回一个值给调用方。函数在 return 语句后会彻底退出
       用途：用于一次性返回结果（值或对象）
yield  功能：yield 用于生成器函数，函数的执行会暂停在 yield 处，并返回一个值给调用方。当调用方再次请求数据时，生成器从上次暂停的位置继续执行（惰性计算）
       用途：用于需要 逐步返回多个值 或 持续生成数据 的场景，例如批量处理数据流
'''

if __name__ == '__main__':
    # 模型保存的位置
    log_dir = './logs/'

    # 打开数据集的 txt
    with open(r'./data/dataset.txt', 'r') as f:
        lines = f.readlines()  # 一次性读取所有行，返回一个列表，每个元素是一行内容

    # 打乱行，这个 txt 主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)  # 设置随机种子，确保随机操作可重复。只要随机种子不变，np.random.shuffle 或 np.random.rand 得到的结果是相同的
    np.random.shuffle(lines)  # 打乱数据
    np.random.seed(None)  # 重置随机数生成器，使后续操作不可预测

    # 90% 数据用于训练，10% 数据用于估计
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立 Alexnet 模型
    model = Alexnet()

    # 保存的方式，3 代 保存一次
    checkpoit_period1 = ModelCheckpoint(
        # 保存文件的路径和命名规则，支持动态变量。例如：ep007-loss0.123-val_loss0.988.h5 （当前epoch，训练集损失值loss，验证集损失值val_loss）
        log_dir + 'ep{epoch:03}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',  # acc: 训练准确率
        save_weights_only=False, # True 表示只保存模型权重，而不是整个模型（含结构）
        save_best_only=True,  # 是否只保存表现最好的模型（False 表示保存所有）
        period=3  # 保存间隔，每 3 个 epoch 保存一次
    )

    # 学习率下降方式，acc 三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc', # 监控的指标，这里是训练集的 acc
        factor=0.5, # 学习率下降的因子，新学习率 = 当前学习率 × factor
        patience=3, # 容忍的 epoch 数，即指标无改善的容忍期
        verbose=1 # 输出详细信息（1 表示打印日志，0 表示不打印）
    )

    '''
    early_stopping 未加入到 model.fit_generator 的 callbacks 回调函数中，因此并不会执行这部分代码去判断是否需要提前停止。
    模型训练会按预设的 epochs=50 完整执行所有轮次
    '''
    # 是否需要早停，当 val_loss 一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss', # 监控的指标，这里是验证集的 val_loss
        min_delta=0, # 最小改善值，低于这个值则认为没有改善
        patience=10, # 容忍的 epoch 数
        verbose=1  # 输出详细信息（1 表示打印日志，0 表示不打印）
    )

    # 配置模型的优化器、损失函数和评估指标，准备进行训练
    # 交叉熵
    model.compile(
        loss='categorical_crossentropy', # 'categorical_crossentropy': 多分类交叉熵，适用于 one-hot 编码的分类任务
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy'] # 评估指标列表，用于在训练和验证时输出额外的性能评估信息
    )

    # 一个批次的大小
    batch_size = 128
    # 打印训练信息
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),
        steps_per_epoch=max(1, num_train // batch_size),  # 每个 epoch 的迭代次数，通常为 num_train // batch_size
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size), # 验证数据生成器
        validation_steps=max(1, num_val // batch_size), # 验证集的迭代次数，通常为 num_val // batch_size
        epochs=50, # 总的epochs
        initial_epoch=0, # 开始训练的 epoch
        callbacks=[checkpoit_period1, reduce_lr] # 回调函数列表，包含 ModelCheckpoint 和 ReduceLROnPlateau
    )
    model.save_weights(log_dir + 'last1.h5')
