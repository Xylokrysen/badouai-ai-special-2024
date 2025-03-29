from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # 定义优化器
        optimizer = Adam(2e-4, 0.5)

        self.discriminator, discriminator_sumarry = self.build_discriminator()
        # 配置模型损失函数和评估指标
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.generator,generator_sumarry = self.build_generator()

        # 生成器以噪声作为输入并生成图像
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # 对于组合模型，我们将仅训练生成器
        self.discriminator.trainable = False

        # 判别器将生成的图像作为输入
        validity = self.discriminator(img)

        # 组合训练（生成器和判别器）
        self.combined = Model(z, validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )

    # 搭建生成器模型
    def build_generator(self):
        # 输入张量
        noise = Input(shape=(self.latent_dim,))

        x = Dense(256)(noise)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(np.prod(self.img_shape), activation='tanh')(x)  # np.prod() 计算所有元素的乘积
        img = Reshape(self.img_shape)(x)

        model = Model(noise, img)
        summary = model.summary()

        return model, summary

    # 搭建判别器模型
    def build_discriminator(self):
        # 输入张量
        img = Input(shape=self.img_shape)

        x = Flatten()(img)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(img, validity)
        summary = model.summary()

        return model, summary

    def train(self, epochs, batch_size=128, sample_interval=50):
        (x_train, _), (_, _) = mnist.load_data()

        # mnist数据集图像像素值的范围0-255，归一化到[-1,1]，符合tanh激活函数
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)  # mnist的图像形状为(n,h,w)，增加一个维度c（灰度图c=1），即 (n,h,w,c) 符合深度学习对数据的要求

        # 生成真假数据标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # 随机取一个批次的图像
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # 生成噪声数据
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 生成假的图像
            gen_imgs = self.generator.predict(noise)

            # 计算判别器的loss
            '''
            train_on_batch 是 Keras 中 Model 对象提供的一个方法，用于在一小批数据上执行一次训练步骤。
            内部自动完成以下操作：
            1. 计算损失（Loss Calculation）：
            根据在编译模型时定义的损失函数，自动计算给定输入数据和对应标签的损失值
            2. 反向传播（Backpropagation）：
            自动计算梯度，并通过优化器更新模型的参数
            
            '''
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # 生成噪声数据
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 训练生成器，让判别器将假图像标签改为1
            g_loss = self.combined.train_on_batch(noise,valid)

            print('%d [D loss: %f,acc.: %.2f%%] [G loss: %f]' % (epoch,d_loss[0],100*d_loss[1],g_loss))

            if epoch%sample_interval==0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))  # 生成一个均值为0，标准差为1 的二维数组 (r*c,self.latent_dim) 数据
        gen_imgs = self.generator.predict(noise)

        # 将-1到1之间的数据归一化数据到 0-1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # fig 表示整个图像的对象
        # axs 是一个二维数组，每个元素对应一个子图
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')  # 在当前子图中显示一张生成的图片，cnt表示第几张图片
                axs[i, j].axis('off')  # 关闭坐标
                cnt += 1
        fig.savefig('./images/mnist_%d.png' % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000,batch_size=32,sample_interval=200)
