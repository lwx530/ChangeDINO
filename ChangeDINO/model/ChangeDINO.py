import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import matplotlib.pyplot as plt
import numpy as np

from .blocks.fpn import FPN, DsBnRelu
from .blocks.cbam import CBAM
from .blocks.adapter import DINOV3Wrapper, LinearAdapter,sal_guide
from .blocks.diffatts import TransformerBlock
from .blocks.refine import LearnableSoftMorph
from .blocks.sfhm import SFHM
from .backbone.mobilenetv2 import mobilenet_v2

class SRFMaskGenerator(nn.Module):
    """
    SRF (Spatial Refinement) 掩码生成器
    作用：从最浅层、物理边界最清晰的 CNN 特征中，提取出一个 0~1 的高清边界权重图。
    """
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


class Detector(nn.Module):
    def __init__(
            self,
            fpn_channels=128,
            n_layers=[1, 1, 1, 1],
            num_classes=1,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

        self.tb5 = nn.Sequential(*[TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree"
        ) for _ in range(n_layers[0])])
        self.tb4 = nn.Sequential(*[TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree"
        ) for _ in range(n_layers[1])])
        self.tb3 = nn.Sequential(*[TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree"
        ) for _ in range(n_layers[2])])
        self.tb2 = nn.Sequential(*[TransformerBlock(
            dim=fpn_channels,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type="BiasFree"
        ) for _ in range(n_layers[3])])

        # 4 个分类头
        self.p5_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p4_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p3_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p2_head = nn.Conv2d(fpn_channels, num_classes, 1)

        # 受 sal_guide 后的分类头 (通道数变化: C+number)
        self.p4_head_g = nn.Conv2d(fpn_channels + 1, num_classes, 1)
        self.p3_head_g = nn.Conv2d(fpn_channels + 2, num_classes, 1)
        self.p2_head_g = nn.Conv2d(fpn_channels + 3, num_classes, 1)

    def _up(self, x):
        return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

    def forward_features(self, xs):
        fea_p2, fea_p3, fea_p4, fea_p5 = xs
        fea_p5 = self.tb5(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, fea_p4)
        fea_p4 = self.tb4(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, fea_p3)
        fea_p3 = self.tb3(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, fea_p2)
        fea_p2 = self.tb2(fea_p2)
        return (fea_p2, fea_p3, fea_p4, fea_p5)  # 都是 [B,128,H,W]

    def forward_heads(self, fea_p2, fea_p3, fea_p4, fea_p5, sal_guides=None):
        # 正常预测
        pred_p5 = self.p5_head(fea_p5)
        pred_p4 = self.p4_head(fea_p4)
        pred_p3 = self.p3_head(fea_p3)
        pred_p2 = self.p2_head(fea_p2)

        # 受 sal_guide 的预测
        guided = {}
        if sal_guides is not None:
            d2_g, d3_g, d4_g = sal_guides
            if d4_g is not None:
                fea_p4_g = sal_guide(fea_p4, d4_g, 1)
                guided['p4'] = self.p4_head_g(fea_p4_g)
            if d3_g is not None:
                fea_p3_g = sal_guide(fea_p3, d3_g, 2)
                guided['p3'] = self.p3_head_g(fea_p3_g)
            if d2_g is not None:
                fea_p2_g = sal_guide(fea_p2, d2_g, 3)
                guided['p2'] = self.p2_head_g(fea_p2_g)

        preds = (self._up(pred_p2), self._up(pred_p3),
                 self._up(pred_p4), self._up(pred_p5))
        guided = {k: self._up(v) for k, v in guided.items()}
        return preds, guided

    def forward(self, xs):
        feats = self.forward_features(xs)
        preds, _ = self.forward_heads(*feats, sal_guides=None)
        return preds

class TwoStageModel(nn.Module):
    def __init__(self, backbone="mobilenetv2", fpn_channels=128,
                 n_layers=[1,1,1,1], **kwargs):
        super().__init__()

        # === Stage 1: DINOv3 粗定位 ===
        self.dino = DINOV3Wrapper(
            weights_path=kwargs.get("dino_weight", "dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
            device="cuda",
            extract_ids=kwargs.get("extract_ids", list(range(24))),
        )
        self.defect_adapter = LinearAdapter(
            in_dim=1024, out_dim=fpn_channels * 2, sizes=(64, 32, 16, 8)
        )
        self.proj_256to128 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels * 2, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])
        self.detector_dino = Detector(
            fpn_channels=fpn_channels, n_layers=n_layers, num_classes=1
        )

        # === Stage 2: MobileNetV2 局部细化 ===
        self.backbone = get_backbone(backbone)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=kwargs.get("deform_groups", 4),
            gamma_mode=kwargs.get("gamma_mode", "SE"),
            beta_mode=kwargs.get("beta_mode", "contextgatedconv"),
        )
        self.sfhm = nn.ModuleList([SFHM(fpn_channels) for _ in range(4)])
        self.detector_mobile = Detector(
            fpn_channels=fpn_channels, n_layers=n_layers, num_classes=1
        )
        self.proj_4to3 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # 初始化为均值权重
        with torch.no_grad():
            self.proj_4to3.weight[:, :3] = torch.eye(3).view(3, 3, 1, 1)  # 保持前3通道不变
            self.proj_4to3.weight[:, 3:] = self.proj_4to3.weight[:, :3].mean(dim=1, keepdim=True) / 3  # 第4通道是前3的均值

        # === SRF 边缘增强 ===
        self.srf = SRFMaskGenerator(fpn_channels)

    def forward(self, x):
        # ========== Stage 1: DINOv3 ==========
        raw_dino = self.dino(x)
        ds_fea = []
        for i in range(4):
            g = raw_dino[i*6:(i+1)*6]
            ds_fea.append(torch.mean(torch.stack(g), dim=0))
        ds_fea = self.defect_adapter(ds_fea)           # 4 × [B,256,8/16/32/64]
        dino_feats = [self.proj_256to128[i](ds_fea[i]) for i in range(4)]

        # DINO 特征过 Detector（只走特征处理，不出预测）
        dino_features = self.detector_dino.forward_features(dino_feats)

        # 出粗预测
        coarse_preds, _ = self.detector_dino.forward_heads(*dino_features, sal_guides=None)
        coarse_main = coarse_preds[0]  # [B,1,256,256]

        # ========== Stage 2: MobileNetV2 细化 ==========
        cnn_in = torch.cat([x, coarse_main], dim=1)
        cnn_feats = self.backbone(self.proj_4to3(cnn_in))
        fpn_feats = self.fpn(cnn_feats[-4:])           # 4 × [B,128,64/32/16/8]
        sfhm_feats = [self.sfhm[i](fpn_feats[i]) for i in range(4)]

        # CNN 特征过 Detector
        cnn_features = self.detector_mobile.forward_features(sfhm_feats)
        refined_preds, _ = self.detector_mobile.forward_heads(*cnn_features, sal_guides=None)
        refined_main = refined_preds[0]                 # [B,1,256,256]

        # ========== SRF 边缘增强（在最大分辨率特征上） ==========
        edge_mask = self.srf(cnn_features[0])           # cnn_features[0] = p2:[B,128,64,64]
        edge_up = F.interpolate(edge_mask, size=(256, 256),
                                mode='bilinear', align_corners=False)

        # ========== Feedback: 细化结果反哺 DINO 解码器 ==========
        sal_guides_dino = [
            F.interpolate(refined_main, size=(64,64), mode='bilinear', align_corners=False),
            F.interpolate(refined_main, size=(32,32), mode='bilinear', align_corners=False),
            F.interpolate(refined_main, size=(16,16), mode='bilinear', align_corners=False),
        ]
        _, guided_preds = self.detector_dino.forward_heads(
            *dino_features, sal_guides=sal_guides_dino
        )

        # 最终预测 = 粗 + 细 + 受指导
        final_pred = coarse_main + refined_main
        for k in ['p2', 'p3', 'p4']:
            if k in guided_preds:
                final_pred = final_pred + guided_preds[k]
        final_pred = final_pred * (1.0 + edge_up * 0.3)

        return final_pred, coarse_preds, refined_preds, guided_preds, edge_mask

    @torch.inference_mode()
    def _forward(self, x):
        final_pred, _, _, _, _ = self.forward(x)
        return final_pred

class ChangeModel(nn.Module):
    def __init__(self, backbone="mobilenetv2", fpn_channels=128,
                 n_layers=[1,1,1,1], **kwargs):
        super().__init__()
        self.model = TwoStageModel(
            backbone=backbone, fpn_channels=fpn_channels,
            n_layers=n_layers, **kwargs
        )

    def forward(self, x):
        return self.model(x)

    @torch.inference_mode()
    def _forward(self, x):
        return self.model._forward(x)