import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .blocks.adapter import DINOV3Wrapper, LinearAdapter
from .backbone.mobilenetv2 import mobilenet_v2
from .blocks.cfsb import CFSB


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


class POE(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_target, f_source, s_gl):
        # 掩码空间对齐并截断置信度
        if s_gl.shape[-2:] != f_source.shape[-2:]:
            s_gl = F.interpolate(s_gl, size=f_source.shape[-2:], mode="bilinear", align_corners=False)
        mask = torch.sigmoid(s_gl)

        # Prototype = E(GAP(S_GL * F_source))
        proto = self.gap(mask * f_source)
        proto_exp = proto.expand_as(f_target)

        # Conv1x1(Cat[F_target, proto])
        return self.conv1x1(torch.cat([f_target, proto_exp], dim=1))


class CPOE(nn.Module):

    def __init__(self, cnn_dim, dino_dim, out_dim):
        super().__init__()
        # 深度可分离卷积调整两路通道至 out_dim
        self.dw_r = DepthwiseSeparableConv(cnn_dim, out_dim)
        self.dw_m = DepthwiseSeparableConv(dino_dim, out_dim)

        # 粗分割预测头 S_GL，同时用于提供逐层深监督
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 1, kernel_size=1)
        )

        self.poe_m2r = POE(out_dim)  # Global -> Local
        self.poe_r2m = POE(out_dim)  # Local -> Global

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_r, f_m):
        # 1. 深度可分离卷积调整通道
        f_dr = self.dw_r(f_r)
        f_dm = self.dw_m(f_m)

        # 尺寸对齐 (以 CNN 分支空间尺寸为基准)
        if f_dm.shape[-2:] != f_dr.shape[-2:]:
            f_dm = F.interpolate(f_dm, size=f_dr.shape[-2:], mode="bilinear", align_corners=False)

        # 2. 生成粗分割掩码 S_GL
        s_gl = self.coarse_conv(torch.cat([f_dr, f_dm], dim=1))

        # 3. 跨原型注入增强 (公式 1 与 2)
        f_rm = self.poe_m2r(f_dr, f_dm, s_gl)
        f_mr = self.poe_r2m(f_dm, f_dr, s_gl)

        f_rm_prime = f_dr + f_dr * f_rm
        f_dm_prime = f_dm + f_dm * f_mr

        # 4. 拼接并通过 1x1 卷积输出 F_CP
        f_cp = self.fuse_conv(torch.cat([f_dm_prime, f_rm_prime], dim=1))
        return f_cp, s_gl


class S_Gate(nn.Module):

    def __init__(self, in_dim, num_gates=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 4, num_gates, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class SSA(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.gate = S_Gate(in_dim, 4)
        self.dw_conv = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1, groups=1, bias=False)
        self.cn2 = nn.Conv2d(in_dim, 1, kernel_size=1, bias=False)
        self.cn3 = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1, bias=False)
        self.cn4 = nn.Conv2d(in_dim, 1, kernel_size=5, padding=2, bias=False)

    def forward(self, x):
        gates = self.gate(x)  # [B, 4, 1, 1]

        m1 = self.dw_conv(x) * gates[:, 0:1]
        m2 = self.cn2(x) * gates[:, 1:2]
        m3 = self.cn3(x) * gates[:, 2:3]
        m4 = self.cn4(x) * gates[:, 3:4]

        return torch.sigmoid(m1 + m2 + m3 + m4)


class SCA(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.gate = S_Gate(in_dim, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 4 个不同压缩比的 MLP (4, 8, 16, 32)
        ratios = [4, 8, 16, 32]
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, max(in_dim // r, 4), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(in_dim // r, 4), in_dim, kernel_size=1)
            ) for r in ratios
        ])

    def forward(self, x):
        gates = self.gate(x)  # [B, 4, 1, 1]
        gap_x = self.gap(x)

        c_att = sum(self.mlps[i](gap_x) * gates[:, i:i + 1] for i in range(4))
        return torch.sigmoid(c_att)


class SSCA(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.ssa = SSA(in_dim)
        self.sca = SCA(in_dim)

    def forward(self, f_m):
        a_s = self.ssa(f_m)
        a_c = self.sca(f_m)
        return f_m + f_m * a_c * a_s


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
        self.cpoe = CPOE(cnn_dim, dino_dim, out_dim)

        self.cfsb = CFSB(in_channels=dino_dim)
        # self.ssca = SSCA(dino_dim)

        self.proj_sc = nn.Sequential(
            nn.Conv2d(dino_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # Cat[F_SC, F_CP, F_GL_deeper] -> out_dim * 3
        self.upsample = UpsampleBlock(in_dim=out_dim * 3, out_dim=out_dim)

    def forward(self, f_r, f_m, f_gl_deeper=None):
        # 1. CPOE 交互提取 F_CP 与深监督粗掩码 s_gl
        f_cp, s_gl = self.cpoe(f_r, f_m)
        target_size = f_cp.shape[-2:]

        # f_sc = self.proj_sc(self.ssca(f_m))
        f_sc = self.proj_sc(self.cfsb(f_m))
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
