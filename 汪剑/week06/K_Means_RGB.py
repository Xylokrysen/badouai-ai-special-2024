import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')

# 转成二维像素
data = img.reshape((-1, 3))
data = np.float32(data)

# 停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means 聚类，分别聚成2、4、8、16、64类
compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
compactness64, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)


# 图像转换回 uint8 二维类型
'''
labels2 每个数据点所属的类别标签，返回的数组形状通常为 (样本数, 1)
centers2：每个聚类中心的坐标（或像素值），通常为浮点数数组，形状为 (2, 特征数)（此处 2 表示聚成 2 类） 特征数 = 通道数
'''
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))


# 图像转换成 RGB 显示
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16,cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64,cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 图像显示
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4', u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
