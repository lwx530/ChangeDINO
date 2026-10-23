import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .blocks.adapter import DINOV3Wrapper, LinearAdapter
from .backbone.mobilenetv2 import mobilenet_v2
from .blocks.bcfm import BCFM, BasicConv2d
# from .blocks.sgda import SGDA
from .blocks.fsd import FSD

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
    def __init__(self, dim=64, use_discrepancy=True):
        super().__init__()
        self.fsd = FSD(dim=dim)
        self.bcfm = BCFM(dim=dim, use_discrepancy=use_discrepancy)

    def forward(self, f_r, f_d):
        # 1. 频域结构解耦
        f_low, f_high = self.fsd(f_r, f_d)
        # 2. 跨频重组与存异补偿
        f_fused, p_aux = self.bcfm(f_low, f_high)
        return f_fused, p_aux


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

        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)
        self.defect_adapter = LinearAdapter(
            in_dim=1024,
            out_dim=fpn_channels,
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

        self.fusion4 = CSDFBlock(dim=dim)
        self.fusion3 = CSDFBlock(dim=dim)
        self.fusion2 = CSDFBlock(dim=dim)
        self.fusion1 = CSDFBlock(dim=dim)

        self.up4 = UpsampleBlock(in_dim=dim, out_dim=dim)
        self.up3 = UpsampleBlock(in_dim=dim * 2, out_dim=dim)
        self.up2 = UpsampleBlock(in_dim=dim * 2, out_dim=dim)
        self.up1 = UpsampleBlock(in_dim=dim * 2, out_dim=dim)

        self.pred_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1)
        )

        self.s1_gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, cnn_feats, dino_feats):
        c1, c2, c3, c4 = cnn_feats
        d1, d2, d3, d4 = dino_feats

        # (最深层, 16x16)
        f_fused4, s4 = self.fusion4(c4, d4)
        f4 = self.up4(f_fused4)

        # (32x32)
        f_fused3, s3 = self.fusion3(c3, d3)
        f_in3 = torch.cat([f_fused3, f4], dim=1)
        f3 = self.up3(f_in3)

        # (64x64)
        f_fused2, s2 = self.fusion2(c2, d2)
        f_in2 = torch.cat([f_fused2, f3], dim=1)
        f2 = self.up2(f_in2)

        # (128x128)
        f_fused1, s1 = self.fusion1(c1, d1)
        f_in1 = torch.cat([f_fused1, f2], dim=1)
        f1 = self.up1(f_in1)

        S_base = self.pred_head(f1)

        if s1.shape[-2:] != S_base.shape[-2:]:
            s1_aligned = F.interpolate(s1, size=S_base.shape[-2:], mode="bilinear", align_corners=False)
        else:
            s1_aligned = s1

        S = S_base + self.s1_gate * s1_aligned

        return S, [s1, s2, s3, s4]

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
