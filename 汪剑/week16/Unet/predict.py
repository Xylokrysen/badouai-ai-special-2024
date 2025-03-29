import cv2
import torch
from week16.Unet.model.unet_model import UNet
import glob
import numpy as np

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = UNet(n_channels=1,n_classes=1)
    net.to(device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth',map_location=device))
    net.eval()

    # 图片处理
    # 读取所有图片
    tests_path = glob.glob('data/test/*.png')
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片，灰度化，转成 (1,1,512,512) 数组，转成torch的tensor格式
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.reshape(1,1,img.shape[0],img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device,dtype=torch.float32)

        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]  # 第一个 [0]：取出第一张图片数据；第二个 [0]：形状 (C, H, W)，取出数据是 (H,W)
        # 处理结果
        pred[pred>=0.5] = 255
        pred[pred<0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path,pred)



