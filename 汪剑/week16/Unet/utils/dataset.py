import torch
import cv2
import os
import glob
from torch.utils.data import Dataset,DataLoader
import random


'''
glob 是 Python 标准库中的一个模块，主要用于查找符合特定规则的文件路径名。
它基于 Unix shell 规则进行文件名匹配（如 *、?、[] 等），可以方便地获取目录下的文件列表
'''

class ISBI_Loader(Dataset):
    def __init__(self,data_path):
        # 初始化函数，读取所有 data_path 下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path,'image/*.png'))

    def augment(self,images,flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(images,flipCode)
        return flip

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        # 根据 image_path 生成 label_path
        label_path = image_path.replace('image','label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转换为单通道
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        image = image.reshape(1,image.shape[0],image.shape[1])
        label = label.reshape(1,label.shape[0],label.shape[1])
        # 处理标签，将像素值为 255 改成 1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为 2 时不做处理
        flipCode = random.choice([-1,0,1,2])  # random.choice() 从 [-1, 0, 1, 2] 这个列表中随机选择一个值赋给 flipCode
        if flipCode != 2:
            image = self.augment(image,flipCode)
            label = self.augment(label,flipCode)
        return image,label

    def __len__(self):
        # 返回数据集大小
        return len(self.imgs_path)

if __name__ == '__main__':
    isbi_loader = ISBI_Loader('../data/train/')
    print('数据个数：',len(isbi_loader))

    train_loader = DataLoader(dataset=isbi_loader,batch_size=2,shuffle=True)
    for image,label in train_loader:
        print(image.shape)
    print(isbi_loader[0])
    print(isbi_loader.__getitem__(0))
