import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_iou import IOU
from pytorch_ssim import SSIM


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.iou = IOU()
        self.ssim = SSIM()

    def forward(self, pred, target):
        # 将 2 通道输出转换为 1 通道的前景概率图
        pred_prob = torch.softmax(pred, dim=1)[:, 1:2]

        target_f = target.float()
        if target_f.dim() == 3:
            target_f = target_f.unsqueeze(1)

        # 1. BCE Loss (对于2通道，CrossEntropy等价于BCE)
        bce = F.cross_entropy(pred, target.squeeze(1).long())
        # 2. IoU Loss
        iou = self.iou(pred_prob, target_f)
        # 3. SSIM Loss
        ssim = 1 - self.ssim(pred_prob, target_f)

        return bce + iou + ssim