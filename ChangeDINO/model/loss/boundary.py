import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 确保pred和target是二值图像
        pred = pred.sigmoid()  # 对预测结果应用Sigmoid，得到概率值
        pred_bin = (pred > 0.5).float()  # 转换为二值图像

        target_bin = target.float()  # 确保target是float类型

        # 使用scipy计算距离变换
        # 由于scipy是基于numpy的，需要先把张量转为numpy数组
        pred_boundary = self.distance_transform_edt(pred_bin)
        target_boundary = self.distance_transform_edt(target_bin)

        # 计算MSE损失
        loss = F.mse_loss(pred_boundary, target_boundary)

        return loss

    def distance_transform_edt(self, tensor):
        """
        对tensor进行距离变换，支持批量处理。
        """
        # 由于scipy处理的是numpy数组，所以需要将tensor转为numpy数组
        batch_size, height, width = tensor.shape
        result = torch.zeros_like(tensor)

        # 对每一张图片进行距离变换
        for i in range(batch_size):
            img = tensor[i].cpu().numpy()  # 转为numpy
            distance_map = distance_transform_edt(img)  # 计算距离变换
            result[i] = torch.tensor(distance_map).to(tensor.device)  # 转回tensor并保留设备

        return result