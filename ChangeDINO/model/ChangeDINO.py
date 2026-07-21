import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import matplotlib.pyplot as plt
import numpy as np

# from .blocks.fpn import FPN, DsBnRelu
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


class DsBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DsBnRelu, self).__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.kernel_size != 1:
            x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GroupWeightFusion(nn.Module):
    def __init__(self, num_groups=4, layers_per_group=6):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_groups, layers_per_group))

    def forward(self, all_feats):  # 24个特征
        outs = []
        for g in range(4):
            group = all_feats[g * 6: (g + 1) * 6]  # 6个[B,N,C]
            w = F.softmax(self.weights[g], dim=-1)  # 可学习权重
            weighted = sum(group[i] * w[i] for i in range(6))
            outs.append(weighted)
        return outs  # 4个group

def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    elif backbone_name == "resnet34":
        backbone = timm.create_model("resnet34", pretrained=False)
        timm.models.load_checkpoint(
            backbone,
            "/root/autodl-tmp/ChangeDINO/model/backbone/resnet34.pth",
            strict=True,
        )
        backbone = timm.create_model("resnet34", pretrained=False, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone
'''elif backbone_name == "resnet34":
        backbone = timm.create_model("resnet34", pretrained=False, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
        timm.models.load_checkpoint(
            backbone,
            "/root/autodl-tmp/ChangeDINO/model/backbone/resnet34.pth",
            strict=False,
        )'''

class ParallelFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=8):
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_branch = DsBnRelu(out_channels, out_channels)

        self.attn_branch = CBAM(out_channels, reduction_ratio)

    def forward(self, x):

        x_reduced = self.reduce(x)

        out_conv = self.conv_branch(x_reduced)

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
            backbone="resnet34",
            fpn_channels=128,
            deform_groups=4,
            gamma_mode="SE",
            beta_mode="contextgatedconv",
            dino_weight="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            device="cuda",
            # extract_ids=[5,11,17,23],
            extract_ids=list(range(24)),
            **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.backbone_channels = self.backbone.channels
        self.cnn_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone_channels[-(i + 1)], fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for i in range(4)
        ])
        '''self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )'''
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)

        self.groupweight = GroupWeightFusion(num_groups=4, layers_per_group=6)

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

        # 实例化 4 个尺度的 SFHM 模块
        '''self.sfhm_modules = nn.ModuleList([
            SFHM(in_dim=fpn_channels) for _ in range(4)
        ])
        # ===============================================================

        dino_adapted_ch = fpn_channels * 2

        self.dino_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dino_adapted_ch, 1, kernel_size=1, bias=True),
                nn.Sigmoid()  # 压缩到 0~1 之间，作为概率权重
            ) for _ in range(4)
        ])'''

    def forward(self, x):

        fea = self.backbone.forward(x)
        # fea = self.fpn(fea[-4:])    # channel：128  size:64,32,16,8
        fea = [self.cnn_proj[3-i](fea[-(4-i)]) for i in range(4)]

        raw_ds_fea = self.dino(x)  # 获取24层

        ds_fea = self.groupweight(raw_ds_fea)

        ds_fea_adapted = self.defect_adapter(ds_fea)

        '''enhanced_feas = []

        for i in range(4):
            sfhm_out = self.sfhm_modules[i](fea[i])
            gate = self.dino_gates[i](ds_fea_adapted[i])
            gated_sfhm_out = sfhm_out * gate
            enhanced_feas.append(gated_sfhm_out)'''

        final_fea = self.pff(fea, ds_fea_adapted)

        return final_fea


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
            **kwargs,
    ):
        super().__init__()

        # 保持门控融合模块
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)
        self.p2_to_p1 = FuseGated(fpn_channels)

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

        self.srf_mask_gen = SRFMaskGenerator(in_channels=fpn_channels)

        self.p4_head = ConvOut(128)
        self.p3_head = ConvOut(128)
        self.p2_head = ConvOut(128)
        self.p1_head = ConvOut(128)

    def forward(self, xs):

        fea1, fea2, fea3, fea4 = xs

        # 从最深层的p4开始
        fea4 = self.tb4(fea4)

        # p5特征融合到p3
        fea3 = self.p4_to_p3(fea4, fea3)
        fea3 = self.tb3(fea3)

        # p4特征融合到p2
        fea2 = self.p3_to_p2(fea3, fea2)
        fea2 = self.tb2(fea2)

        # p3特征融合到p1
        fea1 = self.p2_to_p1(fea2, fea1)
        fea1 = self.tb1(fea1)

        edge_mask = self.srf_mask_gen(fea1)              # B,1,128,128
        fea_p1 = fea1 * (1.0 + edge_mask)

        edge_p2 = F.interpolate(edge_mask, (32, 32), mode='bilinear')
        fea_p2 = fea2 * (1.0 + edge_p2)  # 64×64

        edge_p3 = F.interpolate(edge_mask, (16, 16), mode='bilinear')
        fea_p3 = fea3 * (1.0 + edge_p3)  # 32×32

        edge_p4 = F.interpolate(edge_mask, (8, 8), mode='bilinear')
        fea_p4 = fea4 * (1.0 + edge_p4)

        pred_p4 = self.p4_head(fea_p4)
        pred_p3 = self.p3_head(fea_p3)
        pred_p2 = self.p2_head(fea_p2)
        pred_p1 = self.p1_head(fea_p1)

        # 3. 上采样到统一尺寸
        pred_p1 = F.interpolate(
            pred_p1, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p2 = F.interpolate(
            pred_p2, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=(256, 256), mode="bilinear", align_corners=False
        )

        return pred_p1, pred_p2, pred_p3, pred_p4, edge_mask


class ChangeModel(nn.Module):
    def __init__(self, backbone="resnet34", fpn_channels=128, **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.detector = Detector(fpn_channels=fpn_channels, **kwargs)
        # self.refiner = LearnableSoftMorph(1, 9)

    @torch.inference_mode()
    def _forward(self, x):
        # for inference
        fea = self.encoder(x)
        pred, _, _, _, _ = self.detector(fea)
        # pred = self.refiner(pred)
        return pred

    def forward(self, x):
        # for training
        fea = self.encoder(x)
        pred1, pred2, pred3, pred4, edge_mask = self.detector(fea)
        # final_pred = self.refiner(preds[0])
        return pred1, pred2, pred3, pred4, edge_mask  # pred, pred_p2, pred_p3, pred_p4, pred_p5
