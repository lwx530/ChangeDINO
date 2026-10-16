import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        return self.sigmoid(scale) * x


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class HFFE_CPOE(nn.Module):
    def __init__(self, cnn_dim=64, dino_dim=128, out_dim=64):
        super(HFFE_CPOE, self).__init__()
        self.out_dim = out_dim

        # 1. 深度可分离卷积调整通道至 out_dim (64)
        self.dw_r = nn.Sequential(
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, padding=1, groups=cnn_dim, bias=False),
            nn.BatchNorm2d(cnn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.dw_m = nn.Sequential(
            nn.Conv2d(dino_dim, dino_dim, kernel_size=3, padding=1, groups=dino_dim, bias=False),
            nn.BatchNorm2d(dino_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dino_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 空间注意力机制与卷积提纯 (公式 9, 10)
        self.sam = SpatialAttention(kernel_size=7)
        self.cbr_low = BasicConv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.cbr_high = BasicConv2d(out_dim, out_dim, kernel_size=3, padding=1)

        # 3. 空间权重矩阵生成器 (SWM, 公式 11, 12)
        self.swm_low_conv = nn.Sequential(
            nn.Conv2d(out_dim, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.swm_high_conv = nn.Sequential(
            nn.Conv2d(out_dim, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 4. 坐标注意力机制 (CoordAtt, 公式 15)
        self.coord_att = CoordinateAttention(out_dim * 2, out_dim * 2, reduction=16)

        # 5. 跨尺度残差聚合与最终融合 (公式 16)
        self.cbr1_low = BasicConv2d(out_dim, out_dim, kernel_size=1)
        self.cbr1_high = BasicConv2d(out_dim, out_dim, kernel_size=1)
        self.fuse_conv = BasicConv2d(out_dim * 2, out_dim, kernel_size=1)

        # 6. 辅助粗掩码深监督头（与原 CPOE 保持相同梯度传导逻辑）
        self.coarse_head = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 1, kernel_size=1)
        )

    def forward(self, f_r, f_m):
        # 1. 调整两路分支通道为 out_dim (64)
        f_low = self.dw_r(f_r)   # CNN 局部特征充当 F_low
        f_high = self.dw_m(f_m)  # DINO 全局特征充当 F_high

        # 尺寸对齐防护
        if f_high.shape[-2:] != f_low.shape[-2:]:
            f_high = F.interpolate(f_high, size=f_low.shape[-2:], mode="bilinear", align_corners=False)

        # 2. 空间注意力预处理 (公式 9, 10)
        f_low_prime = self.cbr_low(self.sam(f_low))
        f_high_prime = self.cbr_high(self.sam(f_high))

        # 3. 双向空间权重互校验 (公式 11 ~ 14)
        swm_low = self.swm_low_conv(f_low_prime)
        swm_high = self.swm_high_conv(f_high_prime)

        f_low_double_prime = f_low * swm_high    # DINO 全局先验指导 CNN 局部
        f_high_double_prime = f_high * swm_low   # CNN 边缘指导 DINO 特征

        # 4. 拼接并通过坐标注意力进行水平+垂直方向的空间偏置加权 (公式 15)
        cat_for_ca = torch.cat([f_low_prime, f_high_prime], dim=1)
        ca_weight = torch.sigmoid(self.coord_att(cat_for_ca))

        # 5. 生成深监督粗掩码 s_gl (由两路注意力交互后的表征直接预测)
        s_gl = self.coarse_head(ca_weight)

        # 6. 加权残差汇聚输出 (公式 16)
        w_low, w_high = torch.split(ca_weight, self.out_dim, dim=1)
        out_low = w_low * self.cbr1_low(f_low_prime + f_low_double_prime)
        out_high = w_high * self.cbr1_high(f_high_prime + f_high_double_prime)

        f_cp = self.fuse_conv(torch.cat([out_low, out_high], dim=1))

        return f_cp, s_gl


