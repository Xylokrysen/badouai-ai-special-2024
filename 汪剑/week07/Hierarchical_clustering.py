from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import matplotlib.pyplot as plt

'''
linkage：计算层次聚类的链接矩阵
fcluster：从链接矩阵 Z 中提取聚类标签
dendrogram：绘制树状图（Dendrogram）
'''


'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
'''

'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X,'ward') # ward 方法最小化类间方差（最小化平方误差）
print(Z)

# 从 Z 矩阵生成最终的平面聚类（Flat Clustering）
f = fcluster(Z,4,'distance')
print(f)

fig = plt.figure(figsize=(5,3))
dn = dendrogram(Z)
plt.show()

