import torch
import torch.nn as nn


class CNNGenerator(nn.Module):
    """轻量级的编码器-解码器 CNN，用于图像到图像的回归任务。"""

    def __init__(self, in_channels=3, out_channels=3, feature_dim=64):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

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

