import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    可导形态学边界损失 (Differentiable Morphological Boundary Loss)
    无需 numpy 转换，全 GPU 加速，梯度完美回传，数值严格在 0~1 之间。
    """

    def __init__(self, kernel_size=5):
        super(BoundaryLoss, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, pred, target):
        """
        pred: SRF模块预测的边缘掩码 (edge_mask)，形状 [B, 1, H, W]，已过Sigmoid (0~1)
        target: 真实的缺陷标签，形状 [B, 1, H, W]，(0 和 1)
        """
        target = target.float()

        # 1. 提取真实的缺陷物理边界 (形态学梯度 = 膨胀 - 腐蚀)
        # 用 MaxPool 模拟膨胀 (Dilation)
        target_dilated = F.max_pool2d(
            target, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )
        # 用 负的MaxPool 模拟腐蚀 (Erosion)
        target_eroded = -F.max_pool2d(
            -target, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        )

        # 真实的边缘区域 (1 表示边缘，0 表示非边缘)
        target_boundary = target_dilated - target_eroded

        # 2. 让 SRF 模块预测出的 edge_mask 逼近这个真实的物理边界
        # 此时 pred 和 target_boundary 都在 0~1 之间，MSE 算出来一般只有 0.0x 到 0.x
        loss = F.mse_loss(pred, target_boundary)

        return loss