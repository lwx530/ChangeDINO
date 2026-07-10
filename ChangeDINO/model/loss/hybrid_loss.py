import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_iou import IOU
from pytorch_ssim import SSIM


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.iou = IOU(size_average=True)
        self.ssim = SSIM(window_size=11,size_average=True)
        self.bce = nn.BCELoss(size_average=True)

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        target_f = target.float()
        if target_f.dim() == 3:
            target_f = target_f.unsqueeze(1)

        # 1. BCE Loss (对于2通道，CrossEntropy等价于BCE)
        bce = self.bce(pred_prob, target_f)
        # 2. IoU Loss
        iou = self.iou(pred_prob, target_f)
        # 3. SSIM Loss
        ssim = 1 - self.ssim(pred_prob, target_f)

        return bce + iou + ssim