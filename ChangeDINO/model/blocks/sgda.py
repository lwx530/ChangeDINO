import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDirDWConv(nn.Module):
    """3x3 方核 + 1x5 横向条形核 + 5x1 纵向条形核并联，专为细长划痕/裂纹设计"""
    def __init__(self, dim):
        super().__init__()
        self.dw_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, bias=False)
        self.dw_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.dw_3x3(x) + self.dw_1x5(x) + self.dw_5x1(x)
        return self.act(self.bn(out))


class SGDA(nn.Module):
    """
    Structure-Guided Dynamic Alignment Module (SGDA)
    支持多向条形局部感知与 CNN 边界门控的动态高频保真 (Unsharp Masking)
    """
    def __init__(self, dim=64, kernel_size=3, num_groups=4, use_high_freq=True, use_strip_dw=True):
        super().__init__()
        self.dim = dim
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.num_groups = num_groups
        self.group_channels = dim // num_groups
        self.scale = self.group_channels ** -0.5
        self.use_high_freq = use_high_freq

        # 1. 结构引导 Query 投影
        self.to_q = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        # 2. DINO 全局语义原型 Key 与局部邻域 Value 投影
        self.to_k = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.k, self.k)),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 3. 动态卷积核生成网络
        self.kernel_mlp = nn.Sequential(
            nn.Linear(self.k * self.k, self.k * self.k, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.k * self.k, self.k * self.k, bias=False)
        )

        # 4. 边界门控高频锐化头 (由 CNN 特征 f_r 驱动，alpha 范围 (0, 2))
        if self.use_high_freq:
            self.alpha_head = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, 1, kernel_size=1, bias=True)
            )
            # 负偏置初始化：起步阶段 alpha = 2 * sigmoid(-3.0) ≈ 0.095，训练中自适应增长
            nn.init.constant_(self.alpha_head[-1].bias, -3.0)

        # 5. 局部特征增强分支 (支持切换多向条形卷积或标准 3x3 DWConv)
        if use_strip_dw:
            self.dw_conv = MultiDirDWConv(dim)
        else:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )

        # 6. 输出混合与激活 (开启高频时用 SiLU 保留负向过冲，关闭时退回 ReLU 严格对齐基线)
        self.proj_out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.out_act = nn.SiLU(inplace=True) if self.use_high_freq else nn.ReLU(inplace=True)

    def forward(self, f_r, f_d):
        b, c, h, w = f_d.shape

        # 1. 局部静态/条形分支
        out_local = self.dw_conv(f_d)

        # 2. 跨源生成 Q, K, V
        q = self.to_q(torch.cat([f_r, f_d], dim=1)).view(b, self.num_groups, self.group_channels, h * w)
        k = self.to_k(f_d).view(b, self.num_groups, self.group_channels, self.k * self.k)
        v = self.to_v(f_d)

        # 3. 计算组内亲和度并生成动态低通核
        affinity = torch.einsum('bgcn, bgck -> bgnk', q, k) * self.scale
        dyn_kernel = F.softmax(self.kernel_mlp(affinity), dim=-1)

        # 4. 局部邻域动态低频聚合 (W_low * V)
        v_unfold = F.unfold(v, kernel_size=self.k, padding=self.pad)
        v_unfold = v_unfold.view(b, self.num_groups, self.group_channels, self.k * self.k, h * w)
        dyn_kernel = dyn_kernel.permute(0, 1, 3, 2).unsqueeze(2)
        out_dyn = torch.sum(v_unfold * dyn_kernel, dim=3).reshape(b, c, h, w)

        # 5. 边界门控高频残差注入: out_dyn + alpha * (v - out_dyn)
        if self.use_high_freq:
            alpha = 2.0 * torch.sigmoid(self.alpha_head(f_r))
            out_dyn = out_dyn + alpha * (v - out_dyn)

        # 6. 融合与残差输出
        out = self.proj_out(out_dyn + out_local) + f_d
        return self.out_act(out)