import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([x_avg, x_max], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x2 = torch.cat([x.unsqueeze(2), pattn1.unsqueeze(2)], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        return self.sigmoid(self.pa2(x2))


class HybridLGA_CGA(nn.Module):
    """
    结合 LGA 的跨模态互相抑制（降 MAE/抑噪）与 CGA 的细粒度像素级注意力（提 F/S 指标）
    直接替换原有的 LGA 模块
    """

    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        # 1. 继承 LGA 的双向互校验门控 (CNN 与 DINO 互为先验过滤背景)
        self.gate2_to_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim2, dim1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.gate1_to_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim1, dim2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 通道对齐到 out_dim
        self.proj1 = nn.Conv2d(dim1, out_dim, 1, bias=False) if dim1 != out_dim else nn.Identity()
        self.proj2 = nn.Conv2d(dim2, out_dim, 1, bias=False) if dim2 != out_dim else nn.Identity()

        # 2. 融入 CGA 的三重注意力机制 (替代原有的单通道 SE)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_dim, reduction=8)
        self.pa = PixelAttention(out_dim)

        # 3. 继承 LGA 的强截断 Mixer (含 BatchNorm 和 ReLU，压死背景微小残差，保住 MAE)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1: CNN 特征 [B, dim1, H, W], x2: DINO 特征 [B, dim2, H, W]
        # 第一阶段：LGA 互校验过滤
        g1 = self.gate1_to_2(x1)
        g2 = self.gate2_to_1(x2)
        x1_hat = self.proj1(x1 * (1.0 + g2))
        x2_hat = self.proj2(x2 * (1.0 + g1))

        # 第二阶段：CGA 像素与空间注意力精炼
        initial = x1_hat + x2_hat
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.pa(initial, pattn1)

        # 凸组合动态重加权
        fused = initial + pattn2 * x1_hat + (1.0 - pattn2) * x2_hat

        # 第三阶段：稀疏化与背景置零（压制 MAE 的关键点）
        out = self.channel_mixer(fused)
        return out