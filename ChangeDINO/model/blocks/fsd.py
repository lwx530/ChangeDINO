import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiChannelWavelet(nn.Module):
    """
    2D Haar 小波正变换 (DWT) 与逆变换 (IDWT)
    将输入特征 [B, C, H, W] 无损解耦为 LL(低频主体), LH(水平高频), HL(垂直高频), HH(对角高频)
    """
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        # [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def dwt(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode="reflect")
        weight = self.filters.repeat(C, 1, 1, 1)
        out = F.conv2d(x, weight, stride=2, groups=C)
        out = out.view(B, C, 4, out.shape[-2], out.shape[-1])
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def idwt(self, ll, lh, hl, hh):
        B, C, H_half, W_half = ll.shape
        out = torch.stack([ll, lh, hl, hh], dim=2).view(B, C * 4, H_half, W_half)
        weight = self.filters.repeat(C, 1, 1, 1)
        return F.conv_transpose2d(out, weight, stride=2, groups=C)


class MultiDirStripConv(nn.Module):
    """多向条形残差单元 (对应 SDR 中的高频纹理残差精炼 R_i，专为细长划痕/裂纹设计)"""
    def __init__(self, dim):
        super().__init__()
        self.dw_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, bias=False)
        self.dw_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        res = self.dw_3x3(x) + self.dw_1x5(x) + self.dw_5x1(x)
        res = self.pw(self.act(self.bn(res)))
        return x + res


class FSD(nn.Module):
    """
    Frequency-aware Structural Decoupling (FSD) Module
    结合 DTP 论文 Fig.2(b) 设计：
    1. Multi-Channel Wavelet 解耦出低频子带与方向高频子带，并用可学习向量 alpha (||alpha||_1=1) 重加权
    2. 双支路 FreqAttention (AdaptiveAvgPool 感知全局语义 + AdaptiveMaxPool 捕捉极值裂纹)
    3. 分别输出解耦并精炼后的低频语义特征 f_low 与高频纹理特征 f_high
    """
    def __init__(self, dim=64):
        super().__init__()
        self.wavelet = MultiChannelWavelet()

        # 论文公式: 可学习子带重加权向量 alpha (subject to ||alpha||_1 = 1)
        self.subband_alpha = nn.Parameter(torch.ones(4) / 4.0)

        # 跨源频域初始对齐投影
        self.low_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
        self.high_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )

        # Fig.2(b) 上支路: AdaptiveAvgPool + FreqAttention (低频语义凝聚)
        self.lf_freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Fig.2(b) 下支路: AdaptiveMaxPool + FreqAttention (高频极值响应)
        self.hf_freq_attn = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 高频纹理专精恢复 (嵌入多向条形卷积残差块)
        self.hf_denoise = MultiDirStripConv(dim)

    def forward(self, f_r, f_d):
        """
        f_r: CNN 局部特征 [B, C, H, W] (富含高频边缘但有背景噪点)
        f_d: DINO 全局特征 [B, C, H, W] (富含低频语义但边界模糊)
        """
        H, W = f_r.shape[-2:]

        # 1. 多通道小波分解
        r_ll, r_lh, r_hl, r_hh = self.wavelet.dwt(f_r)
        d_ll, d_lh, d_hl, d_hh = self.wavelet.dwt(f_d)

        # 2. 论文可学习向量 alpha 归一化重加权 (||alpha||_1 = 1)
        alpha = F.softmax(self.subband_alpha, dim=0)

        # 3. 低频子空间 (LL): DINO 与 CNN 低频融合 + AvgPool FreqAttention
        f_ll = self.low_fuse(torch.cat([d_ll, r_ll], dim=1)) * alpha[0]
        f_ll = f_ll * self.lf_freq_attn(f_ll) + d_ll  # 保持 DINO 全局语义基底

        # 4. 高频子空间 (LH, HL, HH): 方向子带加权聚合 + MaxPool FreqAttention + 条形卷积精炼
        r_hf = alpha[1] * r_lh + alpha[2] * r_hl + alpha[3] * r_hh
        d_hf = alpha[1] * d_lh + alpha[2] * d_hl + alpha[3] * d_hh

        f_hf = self.high_fuse(torch.cat([r_hf, d_hf], dim=1))
        f_hf = f_hf * self.hf_freq_attn(f_hf) + r_hf  # MaxPool 强化微弱划痕极值
        f_hf = self.hf_denoise(f_hf)                  # 滤除背景杂波并强化条形连续结构

        # 5. 分离器 (Separator) 还原空间分辨率，输出解耦的双路径特征 {f_low, f_high}
        f_low = F.interpolate(f_ll, size=(H, W), mode="bilinear", align_corners=False) + f_d
        f_high = F.interpolate(f_hf, size=(H, W), mode="bilinear", align_corners=False) + f_r

        return f_low, f_high