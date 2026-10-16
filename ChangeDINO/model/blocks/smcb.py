import torch
import torch.nn as nn
import torch.nn.functional as F


class SMCB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, num_groups=4):
        super(SMCB, self).__init__()
        self.in_channels = in_channels
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups

        # 1. 投影得到 Q, K, V
        self.to_v = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.to_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Key 分支：自适应池化到 (k, k)，提取全局紧凑上下文
        self.to_k_pool = nn.AdaptiveAvgPool2d((self.k, self.k))
        self.to_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 2. 动态核生成网络：将 affinity 映射为各组专属的局部卷积权重
        # 分组预测卷积核，将显存和参数开销进一步压缩
        self.kernel_mlp = nn.Sequential(
            nn.Linear(self.k * self.k, self.k * self.k, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.k * self.k, self.k * self.k, bias=False)
        )

        # 3. 局部特征增强分支 (3x3 Depthwise Conv)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 4. 输出混合
        self.proj_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # ---------- 1. 局部 Depthwise 卷积分支 ----------
        out_local = self.dw_conv(x)

        # ---------- 2. 语义动态核滤波分支 ----------
        v = self.to_v(x)  # [b, c, h, w]

        # 生成 Q: [b, num_groups, group_channels, h*w]
        q = self.to_q(x).view(b, self.num_groups, self.group_channels, h * w)

        # 生成 K (池化到 k*k): [b, num_groups, group_channels, k*k]
        k = self.to_k(self.to_k_pool(x)).view(b, self.num_groups, self.group_channels, self.k * self.k)

        # 计算通道组内的亲和度 Affinity: [b, num_groups, h*w, k*k]
        scale = self.group_channels ** -0.5
        affinity = torch.einsum('bgcn, bgck -> bgnk', q, k) * scale

        # 预测动态核: [b, num_groups, h*w, k*k]
        dyn_kernel = self.kernel_mlp(affinity)
        dyn_kernel = F.softmax(dyn_kernel, dim=-1)  # 对每个位置的 k*k 权重归一化

        # 极低显存展开与加权 (由于 k=3，展平开销极低):
        # [b, c*9, h*w] -> [b, num_groups, group_channels, 9, h*w]
        v_unfold = F.unfold(v, kernel_size=self.k, padding=self.pad)
        v_unfold = v_unfold.view(b, self.num_groups, self.group_channels, self.k * self.k, h * w)

        # 动态核转置对齐: [b, num_groups, 1, 9, h*w]
        dyn_kernel = dyn_kernel.permute(0, 1, 3, 2).unsqueeze(2)

        # 局部动态核加权融合: 对 9 个邻域求和
        out_dyn = torch.sum(v_unfold * dyn_kernel, dim=3)  # [b, num_groups, group_channels, h*w]
        out_dyn = out_dyn.contiguous().view(b, c, h, w)

        # ---------- 3. 融合与残差连接 ----------
        out = self.proj_out(out_dyn + out_local) + x
        return F.relu(out, inplace=True)

