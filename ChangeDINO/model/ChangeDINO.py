import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .blocks.adapter import DINOV3Wrapper, LinearAdapter
from .backbone.mobilenetv2 import mobilenet_v2
from .blocks.hffe import HFFE_CPOE
from .blocks.smcb import SMCB


def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    elif backbone_name == "resnet34":
        backbone = timm.create_model("resnet34", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    return backbone



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



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

    def forward(self, x, target_size=None):
        x = self.convs(x)
        if target_size is not None:
            return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class GLFModule(nn.Module):

    def __init__(self, cnn_dim=64, dino_dim=128, out_dim=64):
        super().__init__()
        self.cpoe = HFFE_CPOE(cnn_dim, dino_dim, out_dim)

        self.smcb = SMCB(in_channels=dino_dim, kernel_size=3, num_groups=4)

        self.proj_sc = nn.Sequential(
            nn.Conv2d(dino_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # Cat[F_SC, F_CP, F_GL_deeper] -> out_dim * 3
        self.upsample = UpsampleBlock(in_dim=out_dim * 3, out_dim=out_dim)

    def forward(self, f_r, f_m, f_gl_deeper=None):
        f_cp, s_gl = self.cpoe(f_r, f_m)
        target_size = f_cp.shape[-2:]

        f_sc = self.proj_sc(self.smcb(f_m))
        if f_sc.shape[-2:] != target_size:
            f_sc = F.interpolate(f_sc, size=target_size, mode="bilinear", align_corners=False)

        # 3. 对齐深层特征 F_GL_deeper
        if f_gl_deeper is None:
            f_gl_deeper = f_cp
        else:
            if f_gl_deeper.shape[-2:] != target_size:
                f_gl_deeper = F.interpolate(f_gl_deeper, size=target_size, mode="bilinear", align_corners=False)

        # 4. 汇聚并上采样 (公式 5)
        f_gl = self.upsample(torch.cat([f_sc, f_cp, f_gl_deeper], dim=1))
        return f_gl, s_gl

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

        # CNN 分支：统一到 fpn_channels (64)
        self.cnn_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone_channels[i], fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for i in range(4)
        ])

        # DINO 分支：统一到 fpn_channels * 2 (128)
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)
        self.defect_adapter = LinearAdapter(
            in_dim=1024,
            out_dim=dense_out_dim,  # 128 通道，完全保留原版设计
            sizes=(128, 64, 32, 16)
        )

    def forward(self, x):
        c_raw = self.backbone(x)
        cnn_feats = [self.cnn_projs[i](c_raw[i]) for i in range(4)]  # 4 个 64 通道特征

        d_raw = self.dino(x)
        dino_feats = self.defect_adapter(d_raw)  # 4 个 128 通道特征

        return cnn_feats, dino_feats


class Decoder(nn.Module):
    def __init__(self, cnn_dim=64, dino_dim=128, out_dim=64):
        super().__init__()
        # 4 个阶段的 GLF 单元 (自顶向下: 16 -> 32 -> 64 -> 128)
        self.glf4 = GLFModule(cnn_dim, dino_dim, out_dim)
        self.glf3 = GLFModule(cnn_dim, dino_dim, out_dim)
        self.glf2 = GLFModule(cnn_dim, dino_dim, out_dim)
        self.glf1 = GLFModule(cnn_dim, dino_dim, out_dim)

        # 论文公式 (6): S = Up(Cat[F_1^GL, F_1^R])
        self.final_up = nn.Sequential(
            nn.Conv2d(out_dim + cnn_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 1, kernel_size=1)
        )

    def forward(self, cnn_feats, dino_feats, raw_img):
        c1, c2, c3, c4 = cnn_feats   # 4 个 64 通道特征
        d1, d2, d3, d4 = dino_feats  # 4 个 128 通道特征

        # 自顶向下逐级解码，收集各层 S_GL 粗掩码
        f_gl4, s4 = self.glf4(c4, d4, f_gl_deeper=None)
        f_gl3, s3 = self.glf3(c3, d3, f_gl_deeper=f_gl4)
        f_gl2, s2 = self.glf2(c2, d2, f_gl_deeper=f_gl3)
        f_gl1, s1 = self.glf1(c1, d1, f_gl_deeper=f_gl2)

        # 对齐至最浅层并拼接 c1 (128x128)
        if f_gl1.shape[-2:] != c1.shape[-2:]:
            f_gl1 = F.interpolate(f_gl1, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        # 最终预测图 S
        out = self.final_up(torch.cat([f_gl1, c1], dim=1))
        S = F.interpolate(out, size=raw_img.shape[-2:], mode="bilinear", align_corners=False)

        return S, [s1, s2, s3, s4]

class ChangeModel(nn.Module):
    def __init__(self, backbone="resnet34", fpn_channels=64, **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.decoder = Decoder(
            cnn_dim=fpn_channels,
            dino_dim=fpn_channels * 2,
            out_dim=fpn_channels
        )

    @torch.inference_mode()
    def _forward(self, x):
        cnn_feats, dino_feats = self.encoder(x)
        S, _ = self.decoder(cnn_feats, dino_feats, raw_img=x)
        return S

    def forward(self, x):
        cnn_feats, dino_feats = self.encoder(x)
        S, coarse_masks = self.decoder(cnn_feats, dino_feats, raw_img=x)
        return S, coarse_masks
