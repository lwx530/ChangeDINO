import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import matplotlib.pyplot as plt
import numpy as np

from .blocks.fpn import FPN, DsBnRelu
from .blocks.cbam import CBAM
from .blocks.adapter import DINOV3Wrapper, LinearAdapter, ConvOut
from .blocks.diffatts import TransformerBlock
from .blocks.refine import LearnableSoftMorph
from .blocks.sfhm import SFHM
from .backbone.mobilenetv2 import mobilenet_v2

class SRFMaskGenerator(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        self.mask_gen = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mask_gen(x)

def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class FuseGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)

class MobileNetv2(nn.Module):
    def __init__(self,
                 backbone="mobilenetv2",
                 fpn_channels=128,
                 **kwargs):
        super(MobileNetv2, self).__init__()
        self.backbone = get_backbone(backbone)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=kwargs.get("deform_groups", 4),
            gamma_mode=kwargs.get("gamma_mode", "SE"),
            beta_mode=kwargs.get("beta_mode", "contextgatedconv"),
        )
        self.sfhm = nn.ModuleList([SFHM(fpn_channels) for _ in range(4)])
        self.tb8 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.tb7 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.tb6 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.tb5 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.p8_to_p7 = FuseGated(fpn_channels)

        self.p7_to_p6 = FuseGated(fpn_channels)

        self.p6_to_p5 = FuseGated(fpn_channels)

        self.p5head = ConvOut(128)

        self.p6head = ConvOut(128 + 1)

        self.p7head = ConvOut(128 + 1)

        self.p8head = ConvOut(128 + 1)

        self.srf = SRFMaskGenerator(fpn_channels)

    def sal_guide(self, feature, sal, number):
        feature_list = torch.chunk(feature, number, 1)
        feature = torch.cat((feature_list[0], sal), 1)
        for i in range(1, number):
            feature = torch.cat((feature, feature_list[i], sal), 1)

        return feature

    def forward(self, x_raw, y1):
        cnn_in = torch.cat([x_raw, y1], dim=1)
        cnn_feats = self.backbone(cnn_in)
        fpn_feats = self.fpn(cnn_feats[-4:])  # 4 × [B,128,64/32/16/8]
        fea5, fea6, fea7, fea8 = [self.sfhm[i](fpn_feats[i]) for i in range(4)]

        feap8 = self.tb8(fea8)
        fea7 = self.p8_to_p7(feap8, fea7)

        feap7 = self.tb7(fea7)
        fea6 = self.p7_to_p6(feap7, fea6)

        feap6 = self.tb6(fea6)       # 32×32
        fea5 = self.p6_to_p5(feap6, fea5)

        feap5 = self.tb5(fea5)

        # SRF执行边缘增强
        edge = self.srf(feap5)
        feap5 = feap5 + feap5 * edge

        out5 = self.p5head(feap5)  # 1，64×64   作为final_pred

        avg_pool = nn.AdaptiveAvgPool2d(32)  # 尺寸下采样到32×32
        out5_1 = avg_pool(out5)
        feap6 = self.sal_guide(feap6, out5_1, 1)
        out6 = self.p6head(feap6)

        avg_pool = nn.AdaptiveAvgPool2d(16)  # 尺寸下采样到16×16
        out5_2 = avg_pool(out5)
        feap7 = self.sal_guide(feap7, out5_2, 1)
        out7 = self.p7head(feap7)

        avg_pool = nn.AdaptiveAvgPool2d(8)  # 尺寸下采样到8×8
        out5_3 = avg_pool(out5)
        feap8 = self.sal_guide(feap8, out5_3, 1)
        out8 = self.p8head(feap8)

        out5 = F.interpolate(out5, size=(256, 256), mode="bilinear", align_corners=False)
        out6 = F.interpolate(out6, size=(256, 256), mode="bilinear", align_corners=False)
        out7 = F.interpolate(out7, size=(256, 256), mode="bilinear", align_corners=False)
        out8 = F.interpolate(out8, size=(256, 256), mode="bilinear", align_corners=False)

        return out5, out6, out7, out8, edge

class TwoStageModel(nn.Module):
    def __init__(self,
                 backbone="mobilenetv2",
                 fpn_channels=128,
                 dino_weight="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                 device="cuda",
                 extract_ids=list(range(24)),
                 **kwargs):
        super().__init__()

        # === Stage 1: DINOv3 粗定位 ===
        self.encoder = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)

        self.defect_adapter = LinearAdapter(
            in_dim=1024, out_dim=fpn_channels, sizes=(64, 32, 16, 8)
        )

        self.tb4 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.tb3 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.tb2 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.tb1 = TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree")

        self.p4_to_p3 = FuseGated(fpn_channels)

        self.p3_to_p2 = FuseGated(fpn_channels)

        self.p2_to_p1 = FuseGated(fpn_channels)

        self.p4head = ConvOut(128+4)

        self.p3head = ConvOut(128+2)

        self.p2head = ConvOut(128+1)

        self.p1head = ConvOut(128)

        self.mobilenet = MobileNetv2()

        # === SRF 边缘增强 ===
        self.srf = SRFMaskGenerator(fpn_channels)

    def sal_guide(self, feature, sal, number):
        feature_list = torch.chunk(feature, number, 1)
        feature = torch.cat((feature_list[0], sal), 1)
        for i in range(1, number):
            feature = torch.cat((feature, feature_list[i], sal), 1)

        return feature

    def forward(self, x):
        # ========== Stage 1: DINOv3 ==========
        feat = self.encoder(x)

        fea1, fea2, fea3, fea4 = self.defect_adapter(feat)

        feap4 = self.tb4(fea4)      # 8,128,8×8
        fea3 = self.p4_to_p3(feap4, fea3)

        feap3 = self.tb3(fea3)      # 8,128,16×16
        fea2 = self.p3_to_p2(feap3, fea2)

        feap2 = self.tb2(fea2)      # 8，128，32×32
        fea1 = self.p2_to_p1(feap2, fea1)

        feap1 = self.tb1(fea1)      # 8，128，64×64

        # SRF执行边缘增强
        edge_mask2 = self.srf(feap1)
        feap1 = feap1 + feap1*edge_mask2

        out1 = self.p1head(feap1)   # 8, 1, 64×64
        out1 = F.interpolate(out1, size=(256, 256), mode="bilinear", align_corners=False)

        outr5, outr6, outr7, outr8, edge_mask1 = self.mobilenet(x, out1)

        avg_pool = nn.AdaptiveAvgPool2d(32)  # 尺寸下采样到32×32
        outr5_1 = avg_pool(outr5)
        feap2 = self.sal_guide(feap2, outr5_1, 1)
        out2 = self.p2head(feap2)

        avg_pool = nn.AdaptiveAvgPool2d(16)  # 尺寸下采样到16×16
        outr5_2 = avg_pool(outr5)
        feap3 = self.sal_guide(feap3, outr5_2, 2)
        out3 = self.p3head(feap3)

        avg_pool = nn.AdaptiveAvgPool2d(8)  # 尺寸下采样到8×8
        outr5_3 = avg_pool(outr5)
        feap4 = self.sal_guide(feap4, outr5_3, 4)
        out4 = self.p4head(feap4)

        out2 = F.interpolate(out2, size=(256, 256), mode="bilinear", align_corners=False)
        out3 = F.interpolate(out3, size=(256, 256), mode="bilinear", align_corners=False)
        out4 = F.interpolate(out4, size=(256, 256), mode="bilinear", align_corners=False)

        return outr5, outr6, outr7, outr8, out1, out2, out3, out4, edge_mask1, edge_mask2

    @torch.inference_mode()
    def _forward(self, x):
        final_pred, _, _, _, _, _, _, _, _, _ = self.forward(x)
        return final_pred

class ChangeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TwoStageModel(**kwargs)

    def forward(self, x):
        return self.model(x)

    @torch.inference_mode()
    def _forward(self, x):
        return self.model._forward(x)