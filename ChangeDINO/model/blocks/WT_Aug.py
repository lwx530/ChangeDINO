import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class DWTForward(nn.Module):
    """离散小波变换（前向）"""

    def __init__(self):
        super(DWTForward, self).__init__()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 确保尺寸为偶数
        if H % 2 != 0 or W % 2 != 0:
            x = F.interpolate(x, size=(H + H % 2, W + W % 2), mode='bilinear', align_corners=False)
            H, W = x.shape[2], x.shape[3]

        # 使用Haar小波进行分解
        x = x.reshape(B * C, 1, H, W)
        LL, (LH, HL, HH) = pywt.dwt2(x.cpu().numpy(), 'haar')

        # 转回tensor
        LL = torch.from_numpy(LL).float().to(x.device)
        LH = torch.from_numpy(LH).float().to(x.device)
        HL = torch.from_numpy(HL).float().to(x.device)
        HH = torch.from_numpy(HH).float().to(x.device)

        # 调整形状
        H2, W2 = LL.shape[2], LL.shape[3]
        LL = LL.reshape(B, C, H2, W2)
        LH = LH.reshape(B, C, H2, W2)
        HL = HL.reshape(B, C, H2, W2)
        HH = HH.reshape(B, C, H2, W2)

        return LL, LH, HL, HH


class DWTInverse(nn.Module):
    """离散小波变换（逆变换）"""

    def __init__(self):
        super(DWTInverse, self).__init__()

    def forward(self, LL, LH, HL, HH):
        # 所有输入: [B, C, H2, W2]
        B, C, H2, W2 = LL.shape

        LL = LL.reshape(B * C, 1, H2, W2)
        LH = LH.reshape(B * C, 1, H2, W2)
        HL = HL.reshape(B * C, 1, H2, W2)
        HH = HH.reshape(B * C, 1, H2, W2)

        # 转为numpy进行逆变换
        LL_np = LL.cpu().numpy()
        LH_np = LH.cpu().numpy()
        HL_np = HL.cpu().numpy()
        HH_np = HH.cpu().numpy()

        coeffs = (LL_np, (LH_np, HL_np, HH_np))
        recon = pywt.idwt2(coeffs, 'haar')

        # 转回tensor
        recon = torch.from_numpy(recon).float().to(LL.device)
        recon = recon.reshape(B, C, H2 * 2, W2 * 2)

        return recon


class WT_Aug(nn.Module):
    """
    小波域特征增强模块 (WT-Aug)
    参考：DINO-AugSeg中的实现
    """

    def __init__(self,
                 mask_prob=0.3,
                 mask_ratio_range=(0.1, 0.5),
                 noise_std=0.05):
        super(WT_Aug, self).__init__()

        self.mask_prob = mask_prob
        self.mask_ratio_range = mask_ratio_range
        self.noise_std = noise_std

        self.dwt_forward = DWTForward()
        self.dwt_inverse = DWTInverse()

        # 可学习的小波系数调制（可选）
        self.alpha_ll = nn.Parameter(torch.ones(1))
        self.alpha_lh = nn.Parameter(torch.ones(1))
        self.alpha_hl = nn.Parameter(torch.ones(1))
        self.alpha_hh = nn.Parameter(torch.ones(1))

    def generate_random_mask(self, shape, device):
        """生成随机掩码"""
        B, C, H, W = shape

        # 随机决定是否应用掩码
        if torch.rand(1).item() > self.mask_prob:
            return torch.ones(shape, device=device)

        # 随机掩码比例
        mask_ratio = torch.empty(1).uniform_(*self.mask_ratio_range).item()

        # 创建随机掩码
        mask = torch.ones(B, C, H, W, device=device)
        num_pixels = H * W
        num_mask = int(num_pixels * mask_ratio)

        for b in range(B):
            for c in range(C):
                # 随机选择要掩码的位置
                flat_mask = mask[b, c].view(-1)
                idx = torch.randperm(num_pixels)[:num_mask]
                flat_mask[idx] = 0

        return mask

    def forward(self, x, training=True):
        """
        x: 输入特征 [B, C, H, W]
        training: 是否在训练模式（仅训练时增强）
        """
        if not training:
            return x

        B, C, H, W = x.shape
        device = x.device

        # 1. 小波分解
        LL, LH, HL, HH = self.dwt_forward(x)

        # 2. 对小波系数进行增强/扰动
        # a) 随机掩码
        mask_ll = self.generate_random_mask(LL.shape, device)
        mask_lh = self.generate_random_mask(LH.shape, device)
        mask_hl = self.generate_random_mask(HL.shape, device)
        mask_hh = self.generate_random_mask(HH.shape, device)

        LL_aug = LL * mask_ll
        LH_aug = LH * mask_lh
        HL_aug = HL * mask_hl
        HH_aug = HH * mask_hh

        # b) 添加轻微噪声（可选）
        if self.noise_std > 0:
            noise_lh = torch.randn_like(LH_aug) * self.noise_std
            noise_hl = torch.randn_like(HL_aug) * self.noise_std
            noise_hh = torch.randn_like(HH_aug) * self.noise_std

            LH_aug = LH_aug + noise_lh
            HL_aug = HL_aug + noise_hl
            HH_aug = HH_aug + noise_hh

        # c) 可学习系数调制
        LL_aug = LL_aug * self.alpha_ll
        LH_aug = LH_aug * self.alpha_lh
        HL_aug = HL_aug * self.alpha_hl
        HH_aug = HH_aug * self.alpha_hh

        # 3. 小波重建
        x_aug = self.dwt_inverse(LL_aug, LH_aug, HL_aug, HH_aug)

        # 4. 确保输出尺寸与输入一致
        if x_aug.shape[2] != H or x_aug.shape[3] != W:
            x_aug = F.interpolate(x_aug, size=(H, W), mode='bilinear', align_corners=False)

        # 5. 残差连接（保持原始信息）
        x_aug = 0.7 * x_aug + 0.3 * x

        return x_aug


class SimpleWT_Aug(nn.Module):
    """
    简化版WT-Aug（如果上面的实现有问题，可以先试这个）
    仅对特征进行小波分解-重建，不做复杂增强
    """

    def __init__(self, augment_prob=0.5, enhance_factor=1.3):
        super(SimpleWT_Aug, self).__init__()
        self.augment_prob = augment_prob  # 增强概率
        self.enhance_factor = enhance_factor  # 增强因子
        self.dwt_forward = DWTForward()
        self.dwt_inverse = DWTInverse()

    def forward(self, x, training=True):
        if not training:
            return x

        # 随机决定是否应用增强
        if torch.rand(1).item() > self.augment_prob:
            return x

        # 小波分解
        LL, LH, HL, HH = self.dwt_forward(x)

        # 对高频成分进行增强
        LH = LH * self.enhance_factor
        HL = HL * self.enhance_factor
        HH = HH * self.enhance_factor

        # 重建
        x_aug = self.dwt_inverse(LL, LH, HL, HH)

        # 调整尺寸
        if x_aug.shape != x.shape:
            x_aug = F.interpolate(x_aug, size=x.shape[2:], mode='bilinear', align_corners=False)

        return x_aug