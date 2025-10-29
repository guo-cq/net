# model.py
import torch
import torch.nn as nn

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    生成一个线性增长的 beta 值序列，共 timesteps 个步骤。
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def get_diffusion_params(timesteps, device):
    """
    根据扩散步数 timesteps，返回 betas、alphas 及其累乘 alphas_cumprod，所有参数转移到指定 device 上。
    """
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas.to(device), alphas.to(device), alphas_cumprod.to(device)

def forward_diffusion_sample(x0, t, alphas_cumprod, device):
    """
    对原始图像 x0 进行正向扩散：
      x_t = sqrt(alphas_cumprod[t]) * x0 + sqrt(1 - alphas_cumprod[t]) * noise
      
    参数:
      x0: 干净图像张量，形状 [B, C, H, W]
      t: 时间步（每个样本的扩散步序），形状 [B]
      alphas_cumprod: 累乘的 alphas 序列（长度为 timesteps）
      device: 计算设备
    返回:
      x_t: 添加噪声后的图像
      noise: 实际添加的噪声
    """
    noise = torch.randn_like(x0).to(device)
    # 将每个样本的 t 对应的参数扩展到 [B, 1, 1, 1]
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t])[:, None, None, None]
    x_t = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
    return x_t, noise

class SimpleNoisePredictor(nn.Module):
    """
    条件噪声预测网络：
      接受三个输入：
        - x: 已加入噪声的干净图像（例如 gt 图像加噪后）
        - t: 扩散时间步（数值条件）
        - cond: 条件图像（例如输入的 "color" 图像）
      
      此处采用简单的拼接方式：将 x 与 cond 在通道上拼接，再拼接经过扩展的 t 信息。
      对于 RGB 图像，x 和 cond 均为3通道，t 扩展为1通道，故最终输入通道数为 7。
    """
    def __init__(self, channels=3):
        super(SimpleNoisePredictor, self).__init__()
        # 输入通道数 = channels (x) + channels (cond) + 1 (t) = 7
        in_channels = channels * 2 + 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t, cond):
        """
        参数:
          x: 带噪图像，形状 [B, 3, H, W]
          t: 扩散时间步，形状 [B]
          cond: 条件图像，形状 [B, 3, H, W]
        返回:
          预测噪声，与 x 通道数一致，形状 [B, 3, H, W]
        """
        # 将 t 扩展为形状 [B, 1, H, W]
        t_emb = t.float().unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [B, 1, 1, 1]
        t_emb = t_emb.repeat(1, 1, x.size(2), x.size(3))           # [B, 1, H, W]
        # 拼接 x、cond 和 t_emb，在通道维度上合并
        input_tensor = torch.cat([x, cond, t_emb], dim=1)  # 形状：[B, 7, H, W]
        out = self.relu(self.conv1(input_tensor))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out
