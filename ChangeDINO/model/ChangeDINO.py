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
from .blocks.sfhm import SFHM
from .backbone.mobilenetv2 import mobilenet_v2

class EdgeExtraction(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        self.edge = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.edge(x)

'''class EdgeExtraction(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        # 1. 常规分支：3x3卷积
        # 兜底捕获各向同性斑点和整体轮廓
        self.branch_normal = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 2. 水平条形分支：1x3卷积
        # 提取垂直走向的细长边缘
        self.branch_horizontal = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 3. 垂直条形分支：3x1卷积
        # 提取水平走向的细长边缘
        self.branch_vertical = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        # 4. 融合与降维层
        # 拼接后通道数为 in_channels * 3，通过 1x1 卷积降维回 in_channels
        # 这里的 1x1 卷积起到了“跨通道注意力”的作用，自适应过滤噪声通道
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 分别提取三种感受野的特征
        edge_normal = self.branch_normal(x)
        edge_h = self.branch_horizontal(x)
        edge_v = self.branch_vertical(x)

        # 在通道维度 (dim=1) 进行拼接 Concat
        # 形状从 [B, C, H, W] 变为 [B, 3C, H, W]
        concat_edge = torch.cat([edge_normal, edge_h, edge_v], dim=1)
        concat_edge = self.relu(concat_edge)

        # 通过 1x1 卷积自适应融合特征并降维回 [B, C, H, W]
        out = self.fusion(concat_edge)

        return out'''

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
        backbone = timm.create_model("resnet34", pretrained=False, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
        state_dict = torch.load(
            "/home/linweixuan/ChangeDINO/model/backbone/resnet34-b627a593.pth",
            map_location="cpu",
            weights_only=True
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        backbone.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


'''class ParallelFusionBlock(nn.Module):
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

        return x1, x2, x3, x4'''


# LGA内部需要的通道重标定模块 (Squeeze-and-Excitation)
class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 适配 CNN (dim1) 与 DINO (dim2) 的 LGA 模块
class LGA(nn.Module):
    def __init__(self, dim1=128, dim2=256, out_dim=128):
        super().__init__()
        # 从模态2生成模态1的门控 (DINO 指导 CNN)
        self.gate2_to_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim2, dim1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # 从模态1生成模态2的门控 (CNN 指导 DINO)
        self.gate1_to_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(dim1, dim2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.se1 = SE_Block(dim1)
        self.se2 = SE_Block(dim2)

        # 论文中的两层 1x1 卷积 Channel Mixer
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim1 + dim2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1: CNN 特征, x2: DINO 特征
        g1 = self.gate1_to_2(x1)
        g2 = self.gate2_to_1(x2)

        # 跨模态门控交互 (残差连接 + 逐元素乘法)
        x1_hat = x1 + g2 * x1
        x2_hat = x2 + g1 * x2

        # 独立SE重标定
        x1_se = self.se1(x1_hat)
        x2_se = self.se2(x2_hat)

        # 拼接与通道降维混合
        out = torch.cat([x1_se, x2_se], dim=1)
        return self.channel_mixer(out)


# 使用 LGA 替换原本的 PFF
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

        # 实例化四个尺度的 LGA 融合模块
        self.lga4 = LGA(dim1=in_dims[3], dim2=hidden_dim, out_dim=in_dims[3])
        self.lga3 = LGA(dim1=in_dims[2], dim2=hidden_dim, out_dim=in_dims[2])
        self.lga2 = LGA(dim1=in_dims[1], dim2=hidden_dim, out_dim=in_dims[1])
        self.lga1 = LGA(dim1=in_dims[0], dim2=hidden_dim, out_dim=in_dims[0])

    def forward(self, feas, ds_fea):
        x1, x2, x3, x4 = feas  # CNN特征: [B, 128, H, W]
        a1, a2, a3, a4 = ds_fea  # DINO特征: [B, 256, H, W]

        # 依次通过各个尺度的双向 LGA 融合
        x4_out = self.lga4(x4, a4)
        x3_out = self.lga3(x3, a3)
        x2_out = self.lga2(x2, a2)
        x1_out = self.lga1(x1, a1)

        return x1_out, x2_out, x3_out, x4_out



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
            extract_ids=[5, 11, 17, 23],
            # extract_ids=list(range(24)),
            **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.backbone_channels = self.backbone.channels
        self.cnn_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone_channels[i], fpn_channels, kernel_size=1, bias=False),
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

        # self.groupweight = GroupWeightFusion(num_groups=4, layers_per_group=6)

        self.defect_adapter = LinearAdapter(
            in_dim=1024,
            out_dim=dense_out_dim,  # 即 256
            sizes=(128, 64, 32, 16)
        )

        self.pff = PyramidFeatureFusion(
            in_dims=[128, 128, 128, 128],
            dense_dim=1024,
            patch_size=16,
            hidden_dim=256,
        )

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
        fea = self.backbone(x)
        # fea = self.fpn(fea[-4:])    # channel：128  size:64,32,16,8
        fea = [self.cnn_proj[i](fea[i]) for i in range(4)]
        # fea = fea[0:4]

        raw_ds_fea = self.dino(x)  # 获取24层

        # ds_fea = self.groupweight(raw_ds_fea)

        ds_fea_adapted = self.defect_adapter(raw_ds_fea)

        enhanced_feas = []

        for i in range(4):
            sfhm_out = self.sfhm_modules[i](fea[i])
            gate = self.dino_gates[i](ds_fea_adapted[i])
            gated_sfhm_out = sfhm_out * gate
            enhanced_feas.append(gated_sfhm_out)

        final_fea = self.pff(enhanced_feas, ds_fea_adapted)

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

        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * fpn_channels, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * fpn_channels, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * fpn_channels, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * fpn_channels, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

        self.edge = EdgeExtraction(in_channels=fpn_channels)

        self.p4_head = ConvOut(128)
        self.p3_head = ConvOut(128)
        self.p2_head = ConvOut(128)
        self.p1_head = ConvOut(128)

        self.conv5 = nn.Conv2d(fpn_channels, 1, kernel_size=1, bias=False)

    def forward(self, xs):

        fea1, fea2, fea3, fea4 = xs

        fea4_up = F.interpolate(fea4, size=(128, 128), mode="bilinear", align_corners=False)
        edge_input = fea1 + fea4_up
        edge_mask = self.edge(edge_input)

        edge_mask_4 = F.interpolate(edge_mask, size=(16, 16), mode="bilinear", align_corners=False)
        fea4E = torch.cat([edge_mask_4, fea4], dim=1)
        t4 = self.conv4(fea4E)
        fea4D = self.tb4(t4)

        edge_mask_3 = F.interpolate(edge_mask, size=(32, 32), mode="bilinear", align_corners=False)
        fea3E = torch.cat([edge_mask_3, fea3], dim=1)
        t3 = self.conv3(fea3E)
        fea3D = self.tb3(self.p4_to_p3(fea4D, t3))

        edge_mask_2 = F.interpolate(edge_mask, size=(64, 64), mode="bilinear", align_corners=False)
        fea2E = torch.cat([edge_mask_2, fea2], dim=1)
        t2 = self.conv2(fea2E)
        fea2D = self.tb2(self.p3_to_p2(fea3D, t2))

        edge_mask_1 = F.interpolate(edge_mask, size=(128, 128), mode="bilinear", align_corners=False)
        fea1E = torch.cat([edge_mask_1, fea1], dim=1)
        t1 = self.conv1(fea1E)
        fea1D = self.tb1(self.p2_to_p1(fea2D, t1))

        '''pred_p4 = self.p4_head(fea4D)
        pred_p3 = self.p3_head(fea3D)
        pred_p2 = self.p2_head(fea2D)'''
        pred_p1 = self.p1_head(fea1D)

        # 3. 上采样到统一尺寸
        pred_p1 = F.interpolate(
            pred_p1, size=(256, 256), mode="bilinear", align_corners=False
        )
        '''pred_p2 = F.interpolate(
            pred_p2, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=(256, 256), mode="bilinear", align_corners=False
        )'''

        edge_mask = self.conv5(edge_mask)

        return pred_p1, edge_mask


class ChangeModel(nn.Module):
    def __init__(self, backbone="resnet34", fpn_channels=128, **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.detector = Detector(fpn_channels=fpn_channels, **kwargs)

    @torch.inference_mode()
    def _forward(self, x):
        # for inference
        final_fea = self.encoder(x)
        pred, _ = self.detector(final_fea)
        return pred

    def forward(self, x):
        # for training
        final_fea = self.encoder(x)
        pred1, edge_mask = self.detector(final_fea)
        return pred1, edge_mask
