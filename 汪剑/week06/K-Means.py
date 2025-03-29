import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png',0)
print(img.shape)
# 获取图像高度和宽度
rows,cols = img.shape[:]

# 将二维图像转成一维图像
data = img.reshape((rows*cols,1))
data = np.float32(data)

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags, centers)
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

# 停止条件 (type,max,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,0.1)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS # 指定 K-Means 算法初始化聚类中心的方法

# K-Means 聚类，聚成 4 类
compactness,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags)
'''
compactness：聚类结果的紧密度（各点到其对应中心距离的平方和），数值越小表示聚类效果越好
labels：每个数据点所属的类别标签（数组），一维结构
centers：每个聚类中心的坐标
'''

# 生成最终的图像
dst = labels.reshape((img.shape[0],img.shape[1]))
# 用来正常显示中文标签
'''
'font.sans-serif'：设置无衬线字体
['SimHei']：指定使用黑体（SimHei）字体，确保图像标题或标签中的中文能够正确显示
'''
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像',u'聚类图像']  # 前缀 u 表示 Unicode 字符串
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1) # 将绘图区域划分为 1 行 2 列的子图区域，并选择第 i+1 个子图
    plt.imshow(images[i],cmap='gray')
    plt.title(titles[i])
    # 去除 x 和 y 轴的刻度，使图像显示更简洁
    plt.xticks([])
    plt.yticks([])
plt.show()
