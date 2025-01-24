import numpy as np
import utils
import cv2
from keras import backend as K
from model.Alexnet import Alexnet

'''
在深度学习中，图像数据有两种格式：
channels_first: (batch, channels, height, width)
channels_last:  (batch, height, width, channels) ---- keras默认格式
'''
K.image_data_format() == 'channels_first'  # 检查图像数据格式是否是 'channels_first'，返回 True 或者 False

if __name__ == '__main__':
    model = Alexnet()
    model.load_weights('./logs/last1.h5')

    img = cv2.imread('test1.jpg')
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255

    '''
    np.expand_dims 用于在指定插入一个新的维度，将数组从低维扩展到高纬。
    由 (h,w,c) → (1,h,w,c) 第一个维度是批次大小，符合神经网络要求的数据
    '''
    img_nor = np.expand_dims(img_nor, axis=0)

    img_resize = utils.resize_image(img_nor, (224, 224))

    '''
    model.predict(img_resize) 输出的是一个概率分布数组。针对本次训练，类似输出 [0.8,0.2]
    np.argmax 获取对应数组中最大值的索引
    '''
    print(model.predict(img_resize))
    print('the answer is: ', utils.print_answer(np.argmax(model.predict(img_resize))))

    cv2.imshow('img: ',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
