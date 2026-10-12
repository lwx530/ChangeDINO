# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pytorch_iou import IOU
# from pytorch_ssim import SSIM
#
#
# class HybridLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.iou = IOU(size_average=True)
#         self.ssim = SSIM(window_size=11,size_average=True)
#         self.bce = nn.BCELoss(size_average=True)
#
#     def forward(self, pred, target):
#         pred_prob = torch.sigmoid(pred)
#         target_f = target.float()
#         if target_f.dim() == 3:
#             target_f = target_f.unsqueeze(1)
#
#         # 1. BCE Loss (对于2通道，CrossEntropy等价于BCE)
#         bce = self.bce(pred_prob, target_f)
#         # 2. IoU Loss
#         iou = self.iou(pred_prob, target_f)
#         # 3. SSIM Loss
#         ssim = 1 - self.ssim(pred_prob, target_f)
#
#         return bce + iou + ssim



import torch
import torch.nn as nn
from pytorch_iou import IOU
from pytorch_ssim import SSIM


class CEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()

        intersection = pred * target

        numerator = (
            (pred - intersection).sum()
            + (target - intersection).sum()
        )

        denominator = pred.sum() + target.sum()

        return numerator / (denominator + self.eps)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.cel = CEL()

        self.bce = nn.BCEWithLogitsLoss()

        self.iou = IOU(size_average=True)
        self.ssim = SSIM(
            window_size=11,
            size_average=True
        )

    def forward(self, pred, target):
        target = target.float()

        if target.dim() == 3:
            target = target.unsqueeze(1)

        # CEL 内部会 sigmoid
        cel_loss = self.cel(pred, target)

        # BCEWithLogitsLoss 直接接收 logits
        bce_loss = self.bce(pred, target)

        # IOU 和 SSIM 使用概率值
        pred_prob = torch.sigmoid(pred)

        iou_loss = self.iou(pred_prob, target)

        ssim_loss = 1 - self.ssim(pred_prob, target)

        total_loss = (
            cel_loss
            + bce_loss
            + iou_loss
            + ssim_loss
        )

        return total_loss