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

        # ==================== 可视化截流代码 ====================
        '''if not self.has_saved_debug:
            os.makedirs('debug_edge_1', exist_ok=True)
            vutils.save_image(target[:8], 'debug_edge_1/1_original_gt.png', normalize=True)
            # 保存这种新方法生成的边缘 GT
            vutils.save_image(target_boundary[:8], 'debug_edge_1/2_gradient_edge_gt.png', normalize=True)
            print("\n📸 [Debug] 梯度法边缘 GT 截获成功！快去对比看看它和之前的形态学边缘有什么区别！")
            self.has_saved_debug = True'''
        # ==============================================================

        # 让 SRF 模块逼近这个完美清晰的梯度边缘
        loss = F.mse_loss(pred, target_boundary)

        return loss

'''class BoundaryLoss(nn.Module):
    """
        可导形态学边界损失 (Differentiable Morphological Boundary Loss)
        无需 numpy 转换，全 GPU 加速，梯度完美回传，数值严格在 0~1 之间。
        """

    def __init__(self, kernel_size=5):
        super(BoundaryLoss, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, pred, target):
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

        return loss'''