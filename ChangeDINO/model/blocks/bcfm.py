import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SWMGenerator(nn.Module):
    """
    三合一空间权重生成器 (合并原 SAM 与 SWM，去除有害的 BatchNorm2d(1))
    同时融合 [通道均值(1ch), 通道极值(1ch), 可学习通道投影(1ch)]，保留细微裂纹的极值响应
    """

    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.mix = nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=True)
        # 负偏置初始化 (-2.0)，使初始掩码均值约为 sigmoid(-2.0) ≈ 0.12 (默认抑制背景)
        nn.init.constant_(self.mix.bias, -2.0)

    def forward(self, x):
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.amax(dim=1, keepdim=True)
        pw_s = self.pw(x)
        return torch.sigmoid(self.mix(torch.cat([avg_s, max_s, pw_s], dim=1)))


class CoordinateAttention(nn.Module):
    """返回纯正交坐标注意力权重 a_w * a_h，范围 (0, 1)"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h


class BCFM(nn.Module):
    """
    Bidirectional Coordinate-Feature Modulation Module (BCFM)
    保留交叉相乘求同主干 + 共享基空间下的非零初始化残差存异补偿
    """

    def __init__(self, dim=64, use_discrepancy=True):
        super().__init__()
        self.dim = dim
        self.use_discrepancy = use_discrepancy

        # 1. 局部特征提纯
        self.cbr_r = BasicConv2d(dim, dim, kernel_size=3, padding=1)
        self.cbr_d = BasicConv2d(dim, dim, kernel_size=3, padding=1)

        # 2. 三合一无 BN 空间权重矩阵生成器
        self.swm_r_gen = SWMGenerator(dim)
        self.swm_d_gen = SWMGenerator(dim)

        # 3. 正交坐标注意力与主干求同融合
        self.coord_att = CoordinateAttention(dim * 2, reduction=16)
        self.cbr1_r = BasicConv2d(dim, dim, kernel_size=1)
        self.cbr1_d = BasicConv2d(dim, dim, kernel_size=1)
        self.fuse_conv = BasicConv2d(dim * 2, dim, kernel_size=1)

        # 4. 残差式分歧补偿分支
        if self.use_discrepancy:
            self.proj_shared = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim)  # 不加 ReLU，保留正负符号信息
            )
            self.dis_head = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim)  # 稳定输出尺度至 O(1)
            )
            # 【关键修复】严禁使用 torch.zeros(1)，必须初始化为非零极小值（如 1e-2），
            # 否则会导致 dis_head 内部权重梯度恒等于 0 产生死锁！
            self.gamma = nn.Parameter(torch.tensor(1e-2))

        # 5. 阶段粗预测深监督头 (位置 A：输出 Logits)
        self.coarse_head = nn.Sequential(
            BasicConv2d(dim, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, 1, kernel_size=1)
        )

    def forward(self, f_r, f_d_align):
        # 1. 特征提纯与空间掩码生成
        f_r_p = self.cbr_r(f_r)
        f_d_p = self.cbr_d(f_d_align)

        swm_r = self.swm_r_gen(f_r_p)
        swm_d = self.swm_d_gen(f_d_p)

        # 2. 双向交叉相乘校验 (求同主干)
        f_r_dp = f_r * swm_d
        f_d_dp = f_d_align * swm_r

        ca_weight = self.coord_att(torch.cat([f_r_p, f_d_p], dim=1))
        w_r, w_d = torch.split(ca_weight, self.dim, dim=1)

        out_r = w_r * self.cbr1_r(f_r_p + f_r_dp)
        out_d = w_d * self.cbr1_d(f_d_p + f_d_dp)
        base = self.fuse_conv(torch.cat([out_r, out_d], dim=1))

        # 3. 共享基空间下的分歧残差补偿 (存异支路)
        if self.use_discrepancy:
            m_dis = (swm_r - swm_d).abs()
            diff = self.proj_shared(f_r) - self.proj_shared(f_d_align)
            f_dis = self.dis_head(torch.cat([diff, diff.abs()], dim=1)) * m_dis
            f_bcfm = base + self.gamma * f_dis

            # 【建议】在训练时可以通过打印或记录这两个值来监控分歧分支是否在正常工作：
            # print(f"gamma: {self.gamma.item():.4f}, m_dis_mean: {m_dis.mean().item():.4f}")
        else:
            f_bcfm = base

        # 4. 生成当前尺度深监督粗预测 (Logits)
        p_aux = self.coarse_head(f_bcfm)

        return f_bcfm, p_aux
