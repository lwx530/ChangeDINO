import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os


class BoundaryLoss(nn.Module):
    """
    基于空间梯度的边界损失 (Gradient-based Boundary Loss)
    完美复刻 MATLAB 的 gradient(gt) 逻辑，专治 1 像素极细划痕！
    """

    def __init__(self):
        super(BoundaryLoss, self).__init__()
        # self.has_saved_debug = True

        # 定义计算 x 方向梯度 (gx) 的 Sobel 卷积核
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)

        # 定义计算 y 方向梯度 (gy) 的 Sobel 卷积核
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).view(1, 1, 3, 3)

        # 注册为 buffer，这样它们会自动随着模型移动到 GPU，且不需要计算梯度
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        """
        pred: SRF模块预测的边缘掩码 (edge_mask) [B, 1, H, W]
        target: 真实的缺陷标签 [B, 1, H, W]
        """
        target = target.float()

        # 1. 对应 MATLAB 的 [gy, gx] = gradient(gt)
        # padding=1 保证卷积后图像尺寸不变
        gx = F.conv2d(target, self.sobel_x, padding=1)
        gy = F.conv2d(target, self.sobel_y, padding=1)

        # 2. 对应 MATLAB 的 temp_edge = gy.*gy + gx.*gx
        temp_edge = gx * gx + gy * gy

        # 3. 对应 MATLAB 的 temp_edge(temp_edge~=0)=1
        # 只要有梯度变化的地方，就是物理边缘
        target_boundary = (temp_edge > 0).float()

        # 让 SRF 模块逼近这个完美清晰的梯度边缘
        # loss = F.mse_loss(pred, target_boundary)
        loss = F.binary_cross_entropy_with_logits(pred, target_boundary)

        return loss