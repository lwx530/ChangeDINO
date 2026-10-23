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
    """三合一空间权重生成器 (对应 CSR 中的空间显著性感知 \phi_SA)"""
    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.mix = nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=True)
        nn.init.constant_(self.mix.bias, -2.0)

    def forward(self, x):
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.amax(dim=1, keepdim=True)
        pw_s = self.pw(x)
        return torch.sigmoid(self.mix(torch.cat([avg_s, max_s, pw_s], dim=1)))


class CoordinateAttention(nn.Module):
    """正交坐标通道注意力 (对应 CSR 中的通道方向感知 \phi_CA)"""
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
    Cross-frequency Bidirectional Coordinate-Feature Modulation (BCFM / CSR)
    融合 DTP 论文 CSR 公式 (3)-(4) 与双向求同存异机制:
    1. 空间分支 \phi_SA: 利用 SWM 进行高低频双向空间交叉调制
    2. 通道坐标分支 \phi_CA: 利用 CoordinateAttention 进行正交方向调制
    3. 动态门控 G_att: 自适应平衡 \phi_CA 与 \phi_SA，并叠加高频残差细节
    """
    def __init__(self, dim=64, use_discrepancy=True):
        super().__init__()
        self.dim = dim
        self.use_discrepancy = use_discrepancy

        # 1. \phi_SA: 空间注意力分支
        self.swm_low = SWMGenerator(dim)
        self.swm_high = SWMGenerator(dim)
        self.sa_fuse = BasicConv2d(dim * 2, dim, kernel_size=1)

        # 2. \phi_CA: 正交坐标通道注意力分支
        self.coord_att = CoordinateAttention(dim * 2, reduction=16)
        self.ca_fuse = BasicConv2d(dim * 2, dim, kernel_size=1)

        # 3. 论文 CSR 公式 (3): 可学习软门控函数 G_att \in [0, 1]
        self.gating = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

        # 4. 频域分歧补偿与高频残差保持 (对应公式 4 的高频结构保留)
        if self.use_discrepancy:
            self.dis_head = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.gamma = nn.Parameter(torch.tensor(1e-2))

        # 5. 深监督粗预测头
        self.coarse_head = nn.Sequential(
            BasicConv2d(dim, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, 1, kernel_size=1)
        )

    def forward(self, f_low, f_high):
        """
        f_low:  FSD 解耦出的低频全局语义特征 [B, C, H, W]
        f_high: FSD 解耦出的高频边缘纹理特征 [B, C, H, W]
        """
        cat_feat = torch.cat([f_low, f_high], dim=1)

        # 1. 计算空间交叉调制特征 \phi_SA(f_low, f_high)
        m_low = self.swm_low(f_low)
        m_high = self.swm_high(f_high)
        phi_sa = self.sa_fuse(torch.cat([f_low * m_high, f_high * m_low], dim=1))

        # 2. 计算坐标通道调制特征 \phi_CA(f_low, f_high)
        ca_weights = self.coord_att(cat_feat)
        phi_ca = self.ca_fuse(cat_feat * ca_weights)

        # 3. 论文公式 (3): G_att 软门控自适应融合
        g_att = self.gating(cat_feat)
        f_csr = g_att * phi_ca + (1.0 - g_att) * phi_sa

        # 4. 论文公式 (4) + 存异补偿: 强化高频边界残差保留
        if self.use_discrepancy:
            m_dis = (m_low - m_high).abs()
            f_bcfm = f_csr + self.gamma * (self.dis_head(f_high) * m_dis)
        else:
            f_bcfm = f_csr

        p_aux = self.coarse_head(f_bcfm)
        return f_bcfm, p_aux