import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet4Layer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet4Layer, self).__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 编码器部分：四个下采样模块
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feature  # 下一个模块的输入通道为当前模块的输出
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True)
        )
        
        # 解码器部分：反向构造上采样模块
        for feature in reversed(features):
            # 上采样部分：转置卷积
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            # 两个卷积层构成解码模块
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 最后用一个1x1卷积映射到输出通道数
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        # 编码器：下采样并保存每一层的特征作为 skip connection
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 倒序排列 skip connection（与解码器对应）
        skip_connections = skip_connections[::-1]
        
        # 解码器：上采样并与对应层的特征拼接
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 上采样
            skip = skip_connections[idx // 2]
            # 若尺寸不一致，则进行插值调整
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            # 拼接
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        
        # 输出层
        return self.final_conv(x)
