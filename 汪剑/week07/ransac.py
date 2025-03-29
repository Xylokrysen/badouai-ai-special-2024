# RANSAC 随机采样一致性（random sample consensus）

import numpy as np
import scipy as sp
import scipy.linalg as sl

'''
scipy: 提供线性代数函数和其它科学计算工具
如：
sp.dot 用于矩阵点乘；
sl.lstsq 用于最小二乘求解
'''


# 辅助函数：随机分割数据
def random_partition(n, n_data):
    '''
    :param n: 生成模型所需的最少样本点
    :param n_data: 样本点数量
    :return:
    '''
    '''
    np.arange() 是 NumPy 中用于生成等差数列数组的函数，类似于 Python 内置的 range()，但返回的是 NumPy 数组，而不是列表
    '''
    all_idxs = np.arange(n_data)  # 生成从 0 到 n_data-1 的索引数组
    np.random.shuffle(all_idxs)  # 将索引打乱
    idxs1 = all_idxs[:n]  # 前 n 个为候选样本索引
    idxs2 = all_idxs[n:]  # 剩余为测试样本索引
    return idxs1, idxs2


class LinearLeastSquareModel:
    # 最小二乘求线性解，用于 RANSAC 的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        '''
        :param input_columns: 数据中作为自变量（x）的列索引
        :param output_columns: 数据中作为因变量（y）的列索引
        :param debug: 参数用于控制是否输出调试信息
        '''
        self.input_columns = input_columns  # 如：input_columns = [0]
        self.output_columns = output_columns  # 如：output_columns = [1]
        self.debug = debug

    def fit(self, data):
        # np.vstack 按垂直方向堆叠数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 返回解向量 x 以及残差、A 的秩和奇异值（行列式接近0）
        x, rediss, rank, s = sl.lstsq(A, B)  # 使用最小二乘法求解 Ax = B，此处省略截距 b
        return x  # 返回斜率 k

    # 计算数据中每个点与模型预测值之间的平方误差
    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, model)  # 计算预测值：B_fit = A * model

        # axis=0 → 沿列方向求和（对每列的所有元素求和，返回每列的一个值）
        # axis=1 → 沿行方向求和（对每行的所有元素求和，返回每行的一个值）
        # axis=None（默认） → 将所有元素求和，返回一个标量值
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 计算每个样本的平方误差
        return err_per_point


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    '''
    :param data: 样本点
    :param model: 假设模型:事先自己确定
    :param n: 生成模型所需的最少样本点
    :param k: 最大迭代次数
    :param t: 阈值:作为判断点满足模型的条件
    :param d: 拟合较好时,需要的样本点最少的个数,当做阈值看待
    :param debug:
    :param return_all:
    :return: bestfit: 最优拟合解（返回nil,如果未找到）
    '''
    '''
    基本思路为：
    1、在数据中随机选择 n 个点作为候选内点；
    2、拟合一个模型；
    3、计算其他点与该模型的误差，将误差低于 t 的点看作内点；
    4、如果内点数超过 d，则用所有内点重新拟合模型，并记录误差；
    5、重复 k 次，选择误差最小的模型作为最终结果。
    '''
    iterations = 0  # 记录迭代次数
    bestfit = None  # 记录当前找到的最佳模型（初始化为 None）
    besterr = np.inf  # 记录最佳模型的误差，初始设为正无穷；越小说明模型拟合效果越好
    best_inliner_idxs = None  # 记录最终内点的索引，用于后续可视化或返回全部信息

    # 迭代过程
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 根据候选索引取出样本数据
        test_points = data[test_idxs]  # 剩余测试点数据
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算测试数据相对于候选模型的误差
        print('test_err = ', test_err < t)  # 打印误差是否小于阈值 t 的布尔数组，便于调试
        also_idxs = test_idxs[test_err < t]  # 筛选测试数据中误差小于 t 的点，作为额外内点（also_idxs）
        print('also_idxs = ', also_idxs)  # 打印额外内点索引
        also_inliers = data[also_idxs, :]  # 从原始数据中提取这些内点
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))

        print('d = ', d)
        '''
        若额外内点的数量超过阈值 d（说明当前模型对大部分数据都适用），则认为这次候选模型较好：
            将候选内点和筛选出的测试内点合并成新的数据 betterdata，并利用它重新拟合模型 bettermodel。
            计算该模型对所有内点的误差，并取均值作为当前模型的评价指标 thiserr。
            若 thiserr 小于记录的最佳误差 besterr，则更新最佳模型、最佳误差以及最佳内点索引
        '''
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 将候选内点与测试内点合并
            bettermodel = model.fit(betterdata)  # # 用所有内点重新拟合模型
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 计算平均误差作为该模型的拟合效果
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 记录所有内点索引
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


'''
测试函数：生成数据、运行 RANSAC 与绘图
'''
def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成 [0, 20) 内的数值作为真实输入
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 生成一个随机线性模型参数（例如斜率），乘以 60 使数值分布较宽
    B_exact = sp.dot(A_exact,perfect_fit)  # 计算完美模型下的输出，即 B = A_exact * perfect_fit

    # 在真实数据上加入高斯噪声，使数据更加接近实际情况
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 添加局外点
    # 随机选择 100 个样本，将其替换为局外点（离群值），以模拟异常情况
    # 对局外点的输入与输出分别重新赋值，使其偏离真实模型
    if 1:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers,n_inputs))
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers,n_outputs))

    # 组合数据
    all_data = np.hstack((A_noisy,B_noisy))  # 形成形如([Xi, Yi])的二维数组，形状 (500, 2)
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]

    # 构造模型实例
    debug = False
    model = LinearLeastSquareModel(input_columns,output_columns,debug=debug)

    # 使用最小二乘法进行全局线性拟合
    linear_fit,resids,rank,s= sl.lstsq(all_data[:,input_columns],all_data[:,output_columns])

    # 运行 RANSAC 算法
    '''
    n=50：每次随机选取 50 个样本作为候选内点。
    k=1000：最大迭代 1000 次。
    t=7e3：误差阈值为 7000，判断一个点是否符合当前模型。
    d=300：至少要求 300 个内点，认为当前模型较好。
    返回的 ransac_fit 为 RANSAC 得到的最佳模型，ransac_data 中包含 'inliers' 键存放所有内点索引
    '''
    ransac_fit,ransac_data = ransac(all_data,model,50,1000,7e3,300,debug=debug,return_all=True)

    if 1:
        import pylab

        # 使用 np.argsort 对真实输入 A_exact 按从小到大排序，便于绘制平滑曲线
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs]

        if 1:
            # 绘制原始带噪数据：用黑色点 'k.' 标记
            pylab.plot(A_noisy[:,0],B_noisy[:,0],'k.',label='data')
            # 绘制 RANSAC 判定为内点的数据：用蓝色叉 'bx' 标记
            pylab.plot(A_noisy[ransac_data['inliers'],0],B_noisy[ransac_data['inliers'],0],'bx',label='RANSAC_data')
        else:
            pylab.plot(A_noisy[non_outlier_idxs,0],B_noisy[non_outlier_idxs,0],'k.',label='noisy data')
            pylab.plot(A_noisy[outlier_idxs,0],B_noisy[outlier_idxs,0],'r.',label='outlier data')

        pylab.plot(A_col0_sorted[:,0],np.dot(A_col0_sorted,ransac_fit)[:,0],label='RANSAC fit')
        pylab.plot(A_col0_sorted[:,0],np.dot(A_col0_sorted,perfect_fit)[:,0],label='exact system')
        pylab.plot(A_col0_sorted[:,0],np.dot(A_col0_sorted,linear_fit)[:,0],label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == '__main__':
    test()
