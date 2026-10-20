import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .blocks.adapter import DINOV3Wrapper, LinearAdapter
from .backbone.mobilenetv2 import mobilenet_v2
from .blocks.bcfm import BCFM, BasicConv2d
from .blocks.sgda import SGDA
# from .blocks.dsrm import DSRM

def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        backbone.channels = [64, 64, 128, 256]
    elif backbone_name == "resnet34":
        backbone = timm.create_model("resnet34", pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        backbone.channels = [64, 64, 128, 256]
    return backbone


class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.convs(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class CSDFBlock(nn.Module):
    """
    Cross-Source Decoding Fusion Block (CSDFBlock)
    支持按层级控制 deeper_dim (消除最深层自引用假拼接) 与 use_high_freq (适配 DINO 原生分辨率)
    """
    def __init__(self, dim=64, deeper_dim=64, use_high_freq=True, use_strip_dw=True, use_discrepancy=True):
        super().__init__()
        self.sgda = SGDA(
            dim=dim, kernel_size=3, num_groups=4,
            use_high_freq=use_high_freq, use_strip_dw=use_strip_dw
        )
        self.bcfm = BCFM(dim=dim, use_discrepancy=use_discrepancy)
        self.upsample = UpsampleBlock(in_dim=dim + deeper_dim, out_dim=dim)

    def forward(self, f_r, f_d, f_deeper=None):
        f_d_align = self.sgda(f_r, f_d)
        f_bcfm, p_aux = self.bcfm(f_r, f_d_align)

        if f_deeper is not None:
            f_in = torch.cat([f_bcfm, f_deeper], dim=1)
        else:
            f_in = f_bcfm

        f_out = self.upsample(f_in)
        return f_out, p_aux


class Encoder(nn.Module):
    def __init__(
            self,
            backbone="resnet34",
            fpn_channels=64,
            dino_weight="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            device="cuda",
            extract_ids=[5, 11, 17, 23],
            **kwargs,
    ):
        super().__init__()
        self.backbone = get_backbone(backbone)
        self.backbone_channels = self.backbone.channels

        self.cnn_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for ch in self.backbone.channels
        ])

        # dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)
        self.defect_adapter = LinearAdapter(
            in_dim=1024,
            out_dim=fpn_channels,  # 128 通道，完全保留原版设计
            sizes=(128, 64, 32, 16)
        )

    def forward(self, x):
        c_raw = self.backbone(x)
        cnn_feats = [self.cnn_projs[i](c_raw[i]) for i in range(4)]  # 4 个 64 通道特征

        d_raw = self.dino(x)
        dino_feats = self.defect_adapter(d_raw)  # 4 个 128 通道特征

        return cnn_feats, dino_feats


class Decoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # csdf4 (16x16): 最深层无 f_deeper (deeper_dim=0)，且为下采样层，关闭高频锐化
        self.csdf4 = CSDFBlock(dim=dim, deeper_dim=0,   use_high_freq=False)
        # csdf3 (32x32): DINOv3 原生 token 分辨率，开启高频锐化
        self.csdf3 = CSDFBlock(dim=dim, deeper_dim=dim, use_high_freq=True)
        # csdf2 (64x64): 2x 适度上采样，开启高频锐化
        self.csdf2 = CSDFBlock(dim=dim, deeper_dim=dim, use_high_freq=True)
        # csdf1 (128x128): 4x 上采样含插值伪影，关闭高频锐化防止放大棋盘格
        self.csdf1 = CSDFBlock(dim=dim, deeper_dim=dim, use_high_freq=False)

        self.pred_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1)
        )

    def forward(self, cnn_feats, dino_feats):
        c1, c2, c3, c4 = cnn_feats
        d1, d2, d3, d4 = dino_feats

        f4, s4 = self.csdf4(c4, d4, f_deeper=None)
        f3, s3 = self.csdf3(c3, d3, f_deeper=f4)
        f2, s2 = self.csdf2(c2, d2, f_deeper=f3)
        f1, s1 = self.csdf1(c1, d1, f_deeper=f2)

        S = self.pred_head(f1)
        return S, [s1, s2, s3, s4]



# class SimpleDecodeBlock(nn.Module):
#     """
#     最简特征融合解码块：
#     1. 将当前尺度的 CNN 特征 (64) 与 DINO 特征 (128) 直接 Concat
#     2. 如果有深层解码特征 (deeper_feat, 64)，也一起 Concat
#     3. 通过两个 3x3 卷积融合并上采样
#     """
#
#     def __init__(self, cnn_dim=64, dino_dim=128, deeper_dim=64, out_dim=64):
#         super().__init__()
#         in_dim = cnn_dim + dino_dim + deeper_dim
#
#         self.fuse = nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True),
#         )
#         # 简单的 1x1 卷积生成深层监督预测图 s_i，保持与 create_ChangeDINO.py 兼容
#         self.aux_head = nn.Conv2d(out_dim, 1, kernel_size=1)
#
#     def forward(self, f_cnn, f_dino, f_deeper=None):
#         target_size = f_cnn.shape[-2:]
#
#         # 1. 对齐 DINO 特征尺寸到当前 CNN 特征尺寸
#         if f_dino.shape[-2:] != target_size:
#             f_dino = F.interpolate(f_dino, size=target_size, mode="bilinear", align_corners=False)
#
#         # 2. 拼接特征
#         if f_deeper is not None:
#             if f_deeper.shape[-2:] != target_size:
#                 f_deeper = F.interpolate(f_deeper, size=target_size, mode="bilinear", align_corners=False)
#             cat_feat = torch.cat([f_cnn, f_dino, f_deeper], dim=1)
#         else:
#             cat_feat = torch.cat([f_cnn, f_dino], dim=1)
#
#         # 3. 卷积融合
#         f_out = self.fuse(cat_feat)
#
#         # 4. 生成当前层级的辅助预测图 (深层监督)
#         s_aux = self.aux_head(f_out)
#
#         return f_out, s_aux
#
#
# class Decoder(nn.Module):
#     def __init__(self, cnn_dim=64, dino_dim=128, out_dim=64):
#         super().__init__()
#         # 第4层（最深层）没有更深层的输入，所以 deeper_dim=0
#         self.dec4 = SimpleDecodeBlock(cnn_dim, dino_dim, deeper_dim=0, out_dim=out_dim)
#         # 第3、2、1层接收上一层的输出，deeper_dim=out_dim
#         self.dec3 = SimpleDecodeBlock(cnn_dim, dino_dim, deeper_dim=out_dim, out_dim=out_dim)
#         self.dec2 = SimpleDecodeBlock(cnn_dim, dino_dim, deeper_dim=out_dim, out_dim=out_dim)
#         self.dec1 = SimpleDecodeBlock(cnn_dim, dino_dim, deeper_dim=out_dim, out_dim=out_dim)
#
#         # 最终预测头：直接把 dec1 的输出映射为单通道预测图
#         self.final_head = nn.Sequential(
#             nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_dim, 1, kernel_size=1)
#         )
#
#     def forward(self, cnn_feats, dino_feats, raw_img):
#         c1, c2, c3, c4 = cnn_feats  # 4个尺度，通道数均为 64
#         d1, d2, d3, d4 = dino_feats  # 4个尺度，通道数均为 128
#
#         # 逐级 Concat + 解码 (从深到浅: 4 -> 3 -> 2 -> 1)
#         f4, s4 = self.dec4(c4, d4, f_deeper=None)
#         f3, s3 = self.dec3(c3, d3, f_deeper=f4)
#         f2, s2 = self.dec2(c2, d2, f_deeper=f3)
#         f1, s1 = self.dec1(c1, d1, f_deeper=f2)
#
#         # 最终输出并恢复到原图尺寸
#         out = self.final_head(f1)
#         S = F.interpolate(out, size=raw_img.shape[-2:], mode="bilinear", align_corners=False)
#
#         return S, [s1, s2, s3, s4]


class SurfDINO(nn.Module):
    def __init__(self, backbone="resnet34", fpn_channels=64, **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.decoder = Decoder(dim=fpn_channels)

    @torch.inference_mode()
    def _forward(self, x):
        cnn_feats, dino_feats = self.encoder(x)
        S, _ = self.decoder(cnn_feats, dino_feats)
        return S

    def forward(self, x):
        cnn_feats, dino_feats = self.encoder(x)
        S, coarse_masks = self.decoder(cnn_feats, dino_feats)
        return S, coarse_masks
