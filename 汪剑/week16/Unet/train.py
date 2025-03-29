import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from week16.Unet.utils.dataset import ISBI_Loader
from week16.Unet.model.unet_model import UNet

def train_net(net,device,data_path,epoch=40,batch_size=1,lr=1e-5):
    # 加载数据集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = DataLoader(dataset=isbi_dataset,batch_size=batch_size,shuffle=True)

    # 定义损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn.to(device)

    # 定义优化器
    optimizer = torch.optim.RMSprop(net.parameters(),lr=lr,weight_decay=1e-8,momentum=0.9)


    writer = SummaryWriter('./logs')

    # 开始训练
    # best_loss 统计，初始化为正无穷
    best_loss = float('inf')
    toal_train_step = 0 # 定义训练次数
    for epoch in range(epoch):
        for data in train_loader:
            # 训练模式
            net.train() # 如果网络中存在一些特殊的层，比如 Dropout、BatchNorm 等等，必须增加这一行代码。一般都加上
            image,label = data
            image = image.to(device=device,dtype=torch.float32)
            label = label.to(device=device,dtype=torch.float32)
            pred = net(image)
            # 计算损失loss
            loss = loss_fn(pred,label)
            # print('Loss/train: {}'.format(loss.item()))
            # 保存 loss 值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(),'best_model.pth')

            optimizer.zero_grad()  # 进行反向传播前先要进行梯度清零
            # 反向传播
            loss.backward() # 获取权重的梯度属性值
            # 优化器更新参数
            optimizer.step()

            toal_train_step += 1
            if toal_train_step % 20 == 0:
                print('训练次数: {}, 损失值: {}'.format(toal_train_step,loss.item()))
                writer.add_scalar('train_step',loss.item(),toal_train_step)


if __name__ == '__main__':
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    unet = UNet(n_channels=1,n_classes=1)
    unet.to(device)
    #开始训练
    train_net(unet,device,data_path='./data/train/')



