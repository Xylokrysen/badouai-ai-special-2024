import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
print(iris)
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(X.shape)  # (150,4)

# plt.scatter(X[:,0],X[:,1],c='red',marker='o',label='see')
# plt.xlabel('speal length')
# plt.ylabel('speal width')
#
# '''
# plt.legend(loc='xxx')
#
# ‘best’（默认值）：自动选择最佳位置。
# ‘upper right’：右上角。
# ‘upper left’：左上角。
# ‘lower right’：右下角。
# ‘lower left’：左下角。
# ‘right’：右侧。
# ‘center left’：左侧中央。
# ‘center right’：右侧中央。
# ‘lower center’：底部中央。
# ‘upper center’：顶部中央。
# '''
# plt.legend(loc=2)  # 图例位置
# plt.show()


dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_
print(label_pred)

# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c='green', marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
