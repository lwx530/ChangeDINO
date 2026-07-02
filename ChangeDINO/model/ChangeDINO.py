import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import matplotlib.pyplot as plt
import numpy as np

from .blocks.fpn import FPN, DsBnRelu
from .blocks.cbam import CBAM
from .blocks.adapter import DINOV3Wrapper, LinearAdapter
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
        # 使用轻量级卷积将 128 维压缩到 1 维
        self.mask_gen = nn.Sequential(
            # 第一层：降维并提取关键边界
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 第二层：压缩为单通道灰度掩码
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            # 关键：Sigmoid 确保输出的乘法权重在 0 到 1 之间
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


import torch
import torch.nn as nn


# 定义一个新的并行融合块
class ParallelFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=8):
        super().__init__()

        # 1. 1x1 卷积降维：只做通道维度的融合，绝对不破坏（模糊）任何空间位置的极细边缘
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 局部卷积分支：负责提取和融合局部空间特征 (保持你原来的 DsBnRelu)
        self.conv_branch = DsBnRelu(out_channels, out_channels)

        # 3. 全局注意力分支：直接在未被 3x3 卷积平滑的特征上寻找异常突变 (保持你原来的 CBAM)
        self.attn_branch = CBAM(out_channels, reduction_ratio)

    def forward(self, x):
        # 先统一降维 (例如 384 -> 128)
        x_reduced = self.reduce(x)

        # ================== 并行双分支 ==================
        # 分支1: 局部平滑与特征融合
        out_conv = self.conv_branch(x_reduced)

        # 分支2: 注意力掩码高亮异常区域
        # 此时的 x_reduced 保留了最原始的高频突变，CBAM 能更准地抓取 MaxPool 和 AvgPool
        out_attn = self.attn_branch(x_reduced)

        return out_conv + out_attn


# 你原本的 PFF 模块，内部替换为并行块
class PyramidFeatureFusion(nn.Module):
    def __init__(
            self,
            in_dims=[128, 128, 128, 128],
            dense_dim=1024,
            patch_size=16,
            hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # 将原来的 nn.Sequential 替换为刚才定义的 ParallelFusionBlock
        self.c4 = ParallelFusionBlock(in_dims[3] + hidden_dim, in_dims[3])
        self.c3 = ParallelFusionBlock(in_dims[2] + hidden_dim, in_dims[2])
        self.c2 = ParallelFusionBlock(in_dims[1] + hidden_dim, in_dims[1])
        self.c1 = ParallelFusionBlock(in_dims[0] + hidden_dim, in_dims[0])

    def forward(self, feas, ds_fea):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_fea  # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        # 前向传播逻辑完全不需要改，依然是先拼接，然后送入融合块
        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4

'''class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128, 128],
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]), CBAM(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]), CBAM(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]), CBAM(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]), CBAM(in_dims[0], 8)
        )

    def forward(self, feas, ds_feas):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_feas  # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4'''


class Encoder(nn.Module):
    def __init__(
            self,
            backbone="mobilenetv2",
            fpn_channels=128,
            deform_groups=4,
            gamma_mode="SE",
            beta_mode="contextgatedconv",
            dino_weight="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            device="cuda",
            # extract_ids=[7, 8, 10, 11, 13, 14, 16, 17],
            extract_ids=list(range(24)),
            **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)

        self.defect_adapter = LinearAdapter(
            in_dim=1024,
            out_dim=dense_out_dim,  # 即 256
            sizes=(64, 32, 16, 8)
        )

        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

        self.srf_mask_gen = SRFMaskGenerator(in_channels=fpn_channels)

        self.fusion_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels + dense_out_dim, fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # 实例化 4 个尺度的 SFHM 模块
        self.sfhm_modules = nn.ModuleList([
            SFHM(in_dim=fpn_channels) for _ in range(4)
        ])
        # ===============================================================

        dino_adapted_ch = fpn_channels * 2

        self.dino_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dino_adapted_ch, 1, kernel_size=1, bias=True),
                nn.Sigmoid()  # 压缩到 0~1 之间，作为概率权重
            ) for _ in range(4)
        ])

    def forward(self, x):

        fea = self.backbone.forward(x)
        fea = self.fpn(fea[-4:])    # channel：128  size:8,16,32,64

        raw_ds_fea = self.dino(x)  # 获取24层

        ds_fea = []
        for i in range(4):
            # 取出当前层的 6 个特征图
            group_feats = raw_ds_fea[i * 6: (i + 1) * 6]
            group_mean_feat = torch.mean(torch.stack(group_feats, dim=0), dim=0)
            ds_fea.append(group_mean_feat)

        ds_fea_adapted = self.defect_adapter(ds_fea)

        enhanced_feas = []

        for i in range(4):
            sfhm_out = self.sfhm_modules[i](fea[i])
            gate = self.dino_gates[i](ds_fea_adapted[i])
            gated_sfhm_out = sfhm_out * gate
            enhanced_feas.append(gated_sfhm_out)

        final_fea = self.pff(enhanced_feas, ds_fea_adapted)

        x1, x2, x3, x4 = final_fea
        edge_mask = self.srf_mask_gen(x1)

        x1_sharpened = x1 * (1.0 + edge_mask)

        return (x1_sharpened, x2, x3, x4), edge_mask


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
            # num_classes=1,
            **kwargs,
    ):
        super().__init__()

        # self.num_classes = num_classes

        # 保持门控融合模块
        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

        self.tb5 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            ) for _ in range(n_layers[0])]
        )
        self.tb4 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            ) for _ in range(n_layers[1])]
        )
        self.tb3 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            ) for _ in range(n_layers[2])]
        )
        self.tb2 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            ) for _ in range(n_layers[3])]
        )
        self.p5_head = nn.Conv2d(fpn_channels, 1, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 1, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 1, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 1, 1)

    def forward(self, xs):

        fea_p2, fea_p3, fea_p4, fea_p5 = xs

        # 2. 自顶向下处理（与原来类似，但没有diff计算）
        # 从最深层的p5开始
        fea_p5 = self.tb5(fea_p5)
        pred_p5 = self.p5_head(fea_p5)

        # p5特征融合到p4
        fea_p4 = self.p5_to_p4(fea_p5, fea_p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)

        # p4特征融合到p3
        fea_p3 = self.p4_to_p3(fea_p4, fea_p3)
        fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)

        # p3特征融合到p2
        fea_p2 = self.p3_to_p2(fea_p3, fea_p2)
        fea_p2 = self.tb2(fea_p2)
        pred_p2 = self.p2_head(fea_p2)

        # 3. 上采样到统一尺寸
        pred_p2 = F.interpolate(
            pred_p2, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p5 = F.interpolate(
            pred_p5, size=(256, 256), mode="bilinear", align_corners=False
        )

        return pred_p2, pred_p3, pred_p4, pred_p5


class ChangeModel(nn.Module):
    def __init__(self, backbone="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1, 1], **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.detector = Detector(fpn_channels=fpn_channels, n_layers=n_layers, **kwargs)
        self.refiner = LearnableSoftMorph(1, 9)

    @torch.inference_mode()
    def _forward(self, x):
        # for inference
        fea, edge_mask = self.encoder(x)
        pred, _, _, _ = self.detector(fea)
        pred = self.refiner(pred)
        return pred

    def forward(self, x):
        # for training
        fea, edge_mask = self.encoder(x)
        preds = self.detector(fea)
        final_pred = self.refiner(preds[0])
        return final_pred, preds, edge_mask  # pred, pred_p2, pred_p3, pred_p4, pred_p5
