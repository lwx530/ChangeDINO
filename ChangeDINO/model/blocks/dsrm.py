# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class MultiDirDWConv(nn.Module):
#     """3x3 方核 + 1x5 横向条形核 + 5x1 纵向条形核并联，提取连续局部结构"""
#     def __init__(self, dim):
#         super().__init__()
#         self.dw_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
#         self.dw_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, bias=False)
#         self.dw_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, bias=False)
#         self.bn = nn.BatchNorm2d(dim)
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.dw_3x3(x) + self.dw_1x5(x) + self.dw_5x1(x)
#         return self.act(self.bn(out))
#
#
# class DSRM(nn.Module):
#     """
#     Dynamic Semantic Refinement Module (DSRM)
#     对 DINO 特征 f_d 进行自精炼：多向条形局部感知 + 动态低频核聚合 + 结构门控高频保真
#     """
#     def __init__(self, dim=64, kernel_size=3, num_groups=4):
#         super().__init__()
#         self.dim = dim
#         self.k = kernel_size
#         self.pad = kernel_size // 2
#         self.num_groups = num_groups
#         self.group_channels = dim // num_groups
#         self.scale = self.group_channels ** -0.5
#
#         # 1. 多向条形局部特征分支
#         self.dw_conv = MultiDirDWConv(dim)
#
#         # 2. Q, K, V 投影
#         self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
#         self.to_k = nn.Sequential(
#             nn.AdaptiveAvgPool2d((self.k, self.k)),
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False)
#         )
#         self.to_v = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim)
#         )
#
#         # 3. 动态卷积核生成网络
#         self.kernel_mlp = nn.Sequential(
#             nn.Linear(self.k * self.k, self.k * self.k, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.k * self.k, self.k * self.k, bias=False)
#         )
#
#         # 4. 连续结构门控的高频保真头 (alpha 范围 (0, 2))
#         self.alpha_head = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, 1, kernel_size=1, bias=True)
#         )
#         nn.init.constant_(self.alpha_head[-1].bias, -3.0)
#
#         # 5. 输出混合与 SiLU 激活 (保留高频负向过冲)
#         self.proj_out = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim)
#         )
#         self.out_act = nn.SiLU(inplace=True)
#
#     def forward(self, f_d):
#         b, c, h, w = f_d.shape
#
#         # 1. 提取多向连续局部结构特征
#         out_local = self.dw_conv(f_d)
#
#         # 2. 生成 Q, K, V
#         q = self.to_q(out_local).view(b, self.num_groups, self.group_channels, h * w)
#         k = self.to_k(f_d).view(b, self.num_groups, self.group_channels, self.k * self.k)
#         v = self.to_v(f_d)
#
#         # 3. 计算组内亲和度并生成动态核
#         affinity = torch.einsum('bgcn, bgck -> bgnk', q, k) * self.scale
#         dyn_kernel = F.softmax(self.kernel_mlp(affinity), dim=-1)
#
#         # 4. 局部邻域动态低频聚合
#         v_unfold = F.unfold(v, kernel_size=self.k, padding=self.pad)
#         v_unfold = v_unfold.view(b, self.num_groups, self.group_channels, self.k * self.k, h * w)
#         dyn_kernel = dyn_kernel.permute(0, 1, 3, 2).unsqueeze(2)
#         out_dyn = torch.sum(v_unfold * dyn_kernel, dim=3).reshape(b, c, h, w)
#
#         # 5. 结构门控高频保真注入
#         alpha = 2.0 * torch.sigmoid(self.alpha_head(out_local))
#         out_dyn = out_dyn + alpha * (v - out_dyn)
#
#         # 6. 融合与残差输出
#         out = self.proj_out(out_dyn + out_local) + f_d
#         return self.out_act(out)



import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDirDWConv(nn.Module):
    """3x3 方核 + 1x5 横向条形核 + 5x1 纵向条形核并联，提取连续局部结构"""
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


class DSRM(nn.Module):
    """
    Dynamic Semantic Refinement Module (DSRM)
    对 DINO 特征 f_d 进行自精炼：多向条形局部感知 + 动态低频核聚合 + 尺度自适应高频保真
    """
    def __init__(self, dim=64, kernel_size=3, num_groups=4, use_high_freq=True):
        super().__init__()
        self.dim = dim
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.num_groups = num_groups
        self.group_channels = dim // num_groups
        self.scale = self.group_channels ** -0.5
        self.use_high_freq = use_high_freq

        # 1. 多向条形局部特征分支
        self.dw_conv = MultiDirDWConv(dim)

        # 2. Q, K, V 投影
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
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

        # 4. 连续结构门控的高频保真头 (仅在原生/轻度上采样尺度开启，alpha 范围 (0, 2))
        if self.use_high_freq:
            self.alpha_head = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, 1, kernel_size=1, bias=True)
            )
            nn.init.constant_(self.alpha_head[-1].bias, -3.0)

        # 5. 输出混合与激活 (开启高频时用 SiLU 保留负向过冲，关闭时用 ReLU)
        self.proj_out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.out_act = nn.SiLU(inplace=True) if self.use_high_freq else nn.ReLU(inplace=True)

    def forward(self, f_d):
        b, c, h, w = f_d.shape

        # 1. 提取多向连续局部结构特征
        out_local = self.dw_conv(f_d)

        # 2. 生成 Q, K, V
        q = self.to_q(f_d).view(b, self.num_groups, self.group_channels, h * w)
        k = self.to_k(f_d).view(b, self.num_groups, self.group_channels, self.k * self.k)
        v = self.to_v(f_d)

        # 3. 计算组内亲和度并生成动态核
        affinity = torch.einsum('bgcn, bgck -> bgnk', q, k) * self.scale
        dyn_kernel = F.softmax(self.kernel_mlp(affinity), dim=-1)

        # 4. 局部邻域动态低频聚合
        v_unfold = F.unfold(v, kernel_size=self.k, padding=self.pad)
        v_unfold = v_unfold.view(b, self.num_groups, self.group_channels, self.k * self.k, h * w)
        dyn_kernel = dyn_kernel.permute(0, 1, 3, 2).unsqueeze(2)
        out_dyn = torch.sum(v_unfold * dyn_kernel, dim=3).reshape(b, c, h, w)

        # 5. 结构门控高频保真注入
        if self.use_high_freq:
            alpha = 2.0 * torch.sigmoid(self.alpha_head(out_local))
            out_dyn = out_dyn + alpha * (v - out_dyn)

        # 6. 融合与残差输出
        out = self.proj_out(out_dyn + out_local) + f_d
        return self.out_act(out)