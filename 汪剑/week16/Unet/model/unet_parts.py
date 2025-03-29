import torch
from torch import nn
from torch.nn import functional as F

'''
nn.ReLU(inplace=True) 解释：
inplace=False（默认）：创建一个新的张量来存储 ReLU 计算结果，不会改变原输入张量。
inplace=True：直接在输入张量上进行修改，节省内存，但可能影响梯度计算
'''


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            '''
            nn.Upsample() 上采样
            参数说明：
            scale_factor: 比例因子，如：scale_factor=2 将输入的高和宽放大 2 倍
            mode: 模式，如：mode='bilinear' 使用双线性插值进行上采样（适用于连续值数据，如图像）
            align_corners: 原始是否对齐角，如：align_corners=True 对齐插值点，以获得更平滑的结果
            '''
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            '''
            nn.ConvTranspose2d 进行转置卷积（反卷积, Transposed Convolution） 上采样

            '''
            nn.ConvTranspose2d(in_channels // 2, out_channels // 2, kernel_size=(2, 2), stride=(2, 2))

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])  # 对应高度H
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])  # 对应宽度W
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # pad = [left, right, top, bottom]  # 适用于2D张量

        x = torch.cat([x1, x2], dim=1)  # 在C维度上进行拼接
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x
