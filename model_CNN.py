import torch
import torch.nn as nn

class CNNGenerator(nn.Module):
    """
    一个简单的 CNN 生成模型，采用编码器－解码器结构：
      - 编码器部分由4个卷积层下采样
      - 解码器部分由4个反卷积上采样
    输出经过 Tanh 激活，范围在 [-1, 1]（你可以根据需要改为 Sigmoid）
    """
    def __init__(self, in_channels=3, out_channels=3, feature_dim=64):
        super(CNNGenerator, self).__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # 输出尺寸：1/2
        self.enc2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim*2),
            nn.ReLU(inplace=True)
        )  # 输出尺寸：1/4
        self.enc3 = nn.Sequential(
            nn.Conv2d(feature_dim*2, feature_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim*4),
            nn.ReLU(inplace=True)
        )  # 输出尺寸：1/8
        self.enc4 = nn.Sequential(
            nn.Conv2d(feature_dim*4, feature_dim*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim*8),
            nn.ReLU(inplace=True)
        )  # 输出尺寸：1/16

        # 解码器
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*8, feature_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim*4),
            nn.ReLU(inplace=True)
        )  # 输出尺寸：1/8
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim*2),
            nn.ReLU(inplace=True)
        )  # 输出尺寸：1/4
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*2, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )  # 输出尺寸：1/2
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )  # 输出尺寸：全尺寸

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x