import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import matplotlib.pyplot as plt
import numpy as np
from .blocks.adapter import DINOV3Wrapper, LinearAdapter, ConvOut
from .blocks.diffatts import TransformerBlock
from .blocks.sfhm import SFHM
from .backbone.mobilenetv2 import mobilenet_v2


# ==================== 1. 2D Haar 小波变换与逆变换 ====================
class DWT_2D(nn.Module):
    """
    可微 2D Haar 小波分解：将特征图分解为低频(LL)与水平(LH)、垂直(HL)、对角高频(HH)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 降采样取样
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # 计算 4 个子带
        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4

        return x_LL, x_LH, x_HL, x_HH


# ==================== 2. 多尺度上下文调制模块 (MSCM) ====================
class MSCM(nn.Module):
    """
    借鉴 WPFormer：利用全局池化与局部空间卷积生成多尺度通道/空间注意力权重，压制高频背景噪声
    """
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g = self.global_branch(self.gap(x))
        l = self.local_branch(x)
        return self.sigmoid(g + l)


# ==================== 3. 小波高低频降噪调制模块 (WCA) ====================
class WaveletContextEnhancer(nn.Module):
    """
    在浅层特征上分解高低频，高频经 MSCM 降噪后重新合成并以残差方式融入
    """
    def __init__(self, dim=128):
        super().__init__()
        self.dwt = DWT_2D()
        self.mscm = MSCM(dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. 2D Haar 分解
        ll, lh, hl, hh = self.dwt(x)
        high = lh + hl + hh  # 聚合多方向高频细节

        # 2. 多尺度上下文调制降噪
        weight = self.mscm(high + ll)
        high_clean = high * weight

        # 3. 频域重构并上采样回原分辨率
        freq_fused = ll + high_clean
        f_up = F.interpolate(freq_fused, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # 4. 残差融合
        out = self.fuse(f_up)
        return x + out


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


class DINOguidedNoiseSuppress(nn.Module):
    def __init__(self, cnn_dim=128, dino_dim=256):
        super().__init__()

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnn_dim + dino_dim, cnn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(cnn_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(cnn_dim, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            for _ in range(4)
        ])

    def forward(self, cnn_feats, dino_feats):
        outs = []

        for i in range(4):
            cnn = cnn_feats[i]
            dino = dino_feats[i]

            if dino.shape[-2:] != cnn.shape[-2:]:
                dino = F.interpolate(
                    dino,
                    size=cnn.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            gate = self.gates[i](torch.cat([cnn, dino], dim=1))

            # 保守抑制：避免一开始把 CNN 特征压没
            clean = cnn * (0.5 + gate)

            outs.append(clean)

        return outs


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

        '''self.noise_suppress = DINOguidedNoiseSuppress(
            cnn_dim=fpn_channels,
            dino_dim=dense_out_dim
        )'''

        # 新增：针对浅层特征（128 与 64 尺度）的小波高频降噪模块
        self.wca1 = WaveletContextEnhancer(fpn_channels)
        self.wca2 = WaveletContextEnhancer(fpn_channels)

    def forward(self, x):
        fea = self.backbone(x)

        fea = [self.cnn_proj[i](fea[i]) for i in range(4)]

        raw_ds_fea = self.dino(x)  # 获取24层

        ds_fea_adapted = self.defect_adapter(raw_ds_fea)

        # fea = self.noise_suppress(fea, ds_fea_adapted)

        final_fea = list(self.pff(fea, ds_fea_adapted))

        final_fea[0] = self.wca1(final_fea[0])
        final_fea[1] = self.wca2(final_fea[1])

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


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


'''class Decoder(nn.Module):
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

        self.convD2 = DecoderConvBlock(fpn_channels, fpn_channels, dilation=2)
        self.convD1 = DecoderConvBlock(fpn_channels, fpn_channels, dilation=1)

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
        fea2D = self.convD2(self.p3_to_p2(fea3D, t2))

        edge_mask_1 = F.interpolate(edge_mask, size=(128, 128), mode="bilinear", align_corners=False)
        fea1E = torch.cat([edge_mask_1, fea1], dim=1)
        t1 = self.conv1(fea1E)
        fea1D = self.convD1(self.p2_to_p1(fea2D, t1))

        pred_p1 = self.p1_head(fea1D)

        # 3. 上采样到统一尺寸
        pred_p1 = F.interpolate(
            pred_p1, size=(256, 256), mode="bilinear", align_corners=False
        )

        edge_mask = self.conv5(edge_mask)

        return pred_p1, edge_mask'''

class Decoder(nn.Module):
    def __init__(self, fpn_channels=128, **kwargs):
        super().__init__()

        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)
        self.p2_to_p1 = FuseGated(fpn_channels)

        self.tb4 = TransformerBlock(dim=fpn_channels, ffn_expansion_factor=2, bias=False, LayerNorm_type="BiasFree")
        self.tb3 = TransformerBlock(dim=fpn_channels, ffn_expansion_factor=2, bias=False, LayerNorm_type="BiasFree")
        self.convD2 = DecoderConvBlock(fpn_channels, fpn_channels, dilation=2)
        self.convD1 = DecoderConvBlock(fpn_channels, fpn_channels, dilation=1)

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

        # 4 个尺度的辅助预测头与 1 个边缘头
        self.p4_head = ConvOut(fpn_channels)
        self.p3_head = ConvOut(fpn_channels)
        self.p2_head = ConvOut(fpn_channels)
        self.p1_head = ConvOut(fpn_channels)
        self.conv5 = nn.Conv2d(fpn_channels, 1, kernel_size=1, bias=False)

    def forward(self, xs):
        fea1, fea2, fea3, fea4 = xs

        # 保持 ESDI-30 边缘构建方式
        fea4_up = F.interpolate(fea4, size=(128, 128), mode="bilinear", align_corners=False)
        edge_input = fea1 + fea4_up
        edge_mask = self.edge(edge_input)

        # Stage 4 (16x16)
        edge_mask_4 = F.interpolate(edge_mask, size=(16, 16), mode="bilinear", align_corners=False)
        fea4E = torch.cat([edge_mask_4, fea4], dim=1)
        t4 = self.conv4(fea4E)
        fea4D = self.tb4(t4)
        pred4 = self.p4_head(fea4D)

        # Stage 3 (32x32)
        edge_mask_3 = F.interpolate(edge_mask, size=(32, 32), mode="bilinear", align_corners=False)
        fea3E = torch.cat([edge_mask_3, fea3], dim=1)
        t3 = self.conv3(fea3E)
        fea3D = self.tb3(self.p4_to_p3(fea4D, t3))
        pred3 = self.p3_head(fea3D)

        # Stage 2 (64x64)
        edge_mask_2 = F.interpolate(edge_mask, size=(64, 64), mode="bilinear", align_corners=False)
        fea2E = torch.cat([edge_mask_2, fea2], dim=1)
        t2 = self.conv2(fea2E)
        fea2D = self.convD2(self.p3_to_p2(fea3D, t2))
        pred2 = self.p2_head(fea2D)

        # Stage 1 (128x128)
        edge_mask_1 = F.interpolate(edge_mask, size=(128, 128), mode="bilinear", align_corners=False)
        fea1E = torch.cat([edge_mask_1, fea1], dim=1)
        t1 = self.conv1(fea1E)
        fea1D = self.convD1(self.p2_to_p1(fea2D, t1))
        pred1 = self.p1_head(fea1D)

        # 统一上采样到 256x256
        pred1 = F.interpolate(pred1, size=(256, 256), mode="bilinear", align_corners=False)
        pred2 = F.interpolate(pred2, size=(256, 256), mode="bilinear", align_corners=False)
        pred3 = F.interpolate(pred3, size=(256, 256), mode="bilinear", align_corners=False)
        pred4 = F.interpolate(pred4, size=(256, 256), mode="bilinear", align_corners=False)

        edge_mask = self.conv5(edge_mask)

        return pred1, pred2, pred3, pred4, edge_mask


class ChangeModel(nn.Module):
    def __init__(self, backbone="resnet34", fpn_channels=128, **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.decoder = Decoder(fpn_channels=fpn_channels, **kwargs)

    @torch.inference_mode()
    def _forward(self, x):
        final_fea = self.encoder(x)
        pred1, _, _, _, _ = self.decoder(final_fea)
        return pred1

    def forward(self, x):
        final_fea = self.encoder(x)
        return self.decoder(final_fea)


'''class ChangeModel(nn.Module):
    def __init__(self, backbone="resnet34", fpn_channels=128, **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.decoder = Decoder(fpn_channels=fpn_channels, **kwargs)

    @torch.inference_mode()
    def _forward(self, x):
        # for inference
        final_fea = self.encoder(x)
        pred, _ = self.decoder(final_fea)
        return pred

    def forward(self, x):
        # for training
        final_fea = self.encoder(x)
        pred1, edge_mask = self.decoder(final_fea)
        return pred1, edge_mask'''
