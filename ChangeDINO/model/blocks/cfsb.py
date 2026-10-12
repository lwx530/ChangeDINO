import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ScharrConv(nn.Module):
    def __init__(self, channel):
        super(ScharrConv, self).__init__()

        # 定义 Scharr 算子的水平和垂直卷积核
        scharr_kernel_x = np.array([[3, 0, -3],
                                    [10, 0, -10],
                                    [3, 0, -3]], dtype=np.float32)

        scharr_kernel_y = np.array([[3, 10, 3],
                                    [0, 0, 0],
                                    [-3, -10, -3]], dtype=np.float32)

        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 注册为 buffer，确保随模型自动转移到 GPU 且不求梯度
        self.register_buffer('scharr_kernel_x', scharr_kernel_x.repeat(channel, 1, 1, 1))
        self.register_buffer('scharr_kernel_y', scharr_kernel_y.repeat(channel, 1, 1, 1))
        self.channel = channel

    def forward(self, x):
        # 使用 functional conv2d，无参数且自动跟随张量设备
        grad_x = nn.functional.conv2d(x, self.scharr_kernel_x, padding=1, groups=self.channel)
        grad_y = nn.functional.conv2d(x, self.scharr_kernel_y, padding=1, groups=self.channel)

        edge_magnitude = grad_x * 0.5 + grad_y * 0.5
        return edge_magnitude


class CFSB(nn.Module):
    def __init__(self, in_channels):
        super(CFSB, self).__init__()

        self.sed = ScharrConv(in_channels)

        # 空间/时域卷积部分
        self.spatial_conv1 = Conv(in_channels, in_channels, k=3)
        self.spatial_conv2 = Conv(in_channels, in_channels, k=3)

        # 频域卷积部分
        self.fft_conv = Conv(in_channels * 2, in_channels * 2, k=1)
        self.fft_conv2 = Conv(in_channels, in_channels, k=3)

        self.final_conv = Conv(in_channels, in_channels, k=3)

    def forward(self, x):
        batch, c, h, w = x.size()

        # 1. 空间域提取
        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        # 2. 频域卷积
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

        fft_feat = self.fft_conv(fft_feat)

        # 3. 还原回空间域
        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        fft_feat = torch.view_as_complex(fft_feat)
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')

        fft_feat = self.fft_conv2(fft_feat)

        # 4. 融合输出
        out = spatial_feat + fft_feat
        return self.final_conv(out)