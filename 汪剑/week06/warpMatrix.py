import numpy as np


'''
assert 语句用于调试和错误检测，在程序执行时检查某个条件是否为真。如果条件为假，则抛出 AssertionError 并停止程序执行

示例：assert condition, "错误提示信息"
当 condition 为 False 时，会抛出异常，并打印提示信息
'''

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix = B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]  # 取出源点 (xi,yi)
        B_i = dst[i, :]  # 取出目标点 (x'i,y'i)

        # [x0, y0, 1, 0, 0, 0, -x0*x'_0, -y0*x'_0] = x'_0
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        # [0, 0, 0, x0, y0, 1, -x0*y'_0, -y0*y'_0] = y'_0
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    '''
    np.array 生成的对象类型为 ndarray
    np.mat 返回的是 matrix 类型，是 ndarray 的子类，专门用于矩阵运算
    '''
    A = np.mat(A) # 将 A 转换为 numpy 矩阵
    warpMatirx = A.I * B  # 计算 A 的逆矩阵 A.I ，求解变换参数

    warpMatirx = np.array(warpMatirx).T[0] # 从 (8,1) 变为 (8,)
    warpMatirx = np.insert(warpMatirx,warpMatirx.shape[0],values=1,axis=0)  # 在索引 8 处插入 1.0，变为 (9,)
    warpMatirx = np.reshape(warpMatirx,(3,3))
    return warpMatirx


if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]

    src = np.array(src)
    dst = np.array(dst)

    warpMatirx = WarpPerspectiveMatrix(src,dst)
    print('warpMatirx:\n',warpMatirx)

