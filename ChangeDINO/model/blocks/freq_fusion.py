
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hamming2D(M, N):
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FreqFusionBlock(nn.Module):
    """
    针对 2D 显著性缺陷检测定制的高低频感知跨层融合模块 (FreqFusion 核心思想)
    hr_feat: 浅层高分辨率特征 (High Resolution)
    lr_feat: 深层低分辨率特征 (Low Resolution)
    """

    def __init__(self, channels=128, lowpass_kernel=5, highpass_kernel=3):
        super().__init__()
        self.channels = channels
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel

        # 1. 通道压缩与交互映射
        mid_dim = channels // 2
        self.hr_proj = nn.Conv2d(channels, mid_dim, kernel_size=1, bias=False)
        self.lr_proj = nn.Conv2d(channels, mid_dim, kernel_size=1, bias=False)

        # 2. 高频与低频残差掩码生成器
        self.low_encoder = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.high_encoder = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 3. 高通滤波提取 (拉普拉斯算子 / 高通差分提取高频细节)
        laplacian_kernel = torch.tensor([[0., 1., 0.],
                                         [1., -4., 1.],
                                         [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel.repeat(channels, 1, 1, 1))

        # 4. 融合后的特征混合 (替代原本 FuseGated 中的 mix)
        self.out_mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, lr_feat, hr_feat):
        """
        lr_feat: 深层低分辨率输入 [B, C, H/2, W/2] (例如 fea4D, fea3D, fea2D)
        hr_feat: 浅层高分辨率输入 [B, C, H, W] (例如 t3, t2, t1)
        """
        # 1. 基础双线性插值上采样
        lr_up = F.interpolate(lr_feat, size=hr_feat.shape[-2:], mode="bilinear", align_corners=False)

        # 2. 浅层高分辨率特征的高频细节提取 (边界与细微划痕响应)
        hr_high_pass = F.conv2d(hr_feat, self.laplacian_kernel, padding=1, groups=self.channels)

        # 3. 联合表征与高低频自适应动态加权
        lr_comp = self.lr_proj(lr_up)
        hr_comp = self.hr_proj(hr_feat)

        low_gate = self.low_encoder(lr_comp + hr_comp)
        high_gate = self.high_encoder(lr_comp + hr_comp)

        # 4. 高频补偿上采样特征，低频稳定主体语义
        fused = hr_feat + low_gate * lr_up + high_gate * hr_high_pass

        return self.out_mix(fused)