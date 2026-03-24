import torch
import torch.nn as nn
import torch.nn.functional as F
import re

REPO_DIR = "dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12,
    "VITSP": 12,
    "VITB": 12,
    "VITL": 24,
    "VITHP": 32,
    "VIT7B": 40,
}

# 从DiveSeg复制的组件
'''class StyleInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_a = nn.Linear(dim, dim)
        self.atten = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)

    def forward(self, x, a):
        cls = x[:, :1, :]  # extract CLS token
        x = x[:, 1:, :]
        a_proj = self.proj_a(a)  # (bs, 1, dim)
        x, _ = self.atten(x, a_proj, a_proj)
        return torch.cat([cls, x], dim=1), a_proj  # restore CLS; return projected style token

class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim)
        self.active = nn.GELU()
        self.proj_2 = nn.Linear(dim, dim)

    def forward(self, x):
        cls = x[:, :1, :]  # extract CLS token
        x = x[:, 1:, :]
        x = self.proj_1(x)  # (bs, n, dim)
        x = self.active(x)
        x = self.proj_2(x)
        return torch.cat([cls, x], dim=1)  # restore CLS to the front

class StyleExtractor(nn.Module):
    def __init__(self, em_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(3, em_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(em_dim, em_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(em_dim, em_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images_a):
        x = self.conv1(images_a)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        bs, dim, _, _ = x.shape
        x = x.view(bs, dim, -1).transpose(1, 2)
        return x'''


class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        # weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        )
        self.model = self.model.eval().to(device)
        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = extract_ids

        # freeze the backbone
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    x, n=range(self.n_layers), reshape=True, norm=True
                )
                feats_ = []
                for i in range(len(self.extract_ids)):
                    feats_.append(feats[self.extract_ids[i]])  # [B, N, C]
        return feats_


'''class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        # weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
        use_aquastyle=False,  # 新增参数
    ):
        super().__init__()
        self.device = device
        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        )
        self.model = self.model.eval().to(device)
        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = extract_ids
        self.use_aquastyle = use_aquastyle

        # freeze the backbone
        for p in self.model.parameters():
            p.requires_grad = False

        # 添加AquaStyle组件
        if use_aquastyle:
            self.feature_dim = self.model.embed_dim
            self.style_extractor = StyleExtractor(em_dim=self.feature_dim)

            # 为每个提取层创建组件
            num_layers = len(extract_ids)
            self.style_injections = nn.ModuleList([
                StyleInjection(dim=self.feature_dim) for _ in range(num_layers)
            ])
            self.adapters = nn.ModuleList([
                Adapter(dim=self.feature_dim) for _ in range(num_layers)
            ])

            # 让AquaStyle组件可训练
            for module in [self.style_extractor] + list(self.style_injections) + list(self.adapters):
                for p in module.parameters():
                    p.requires_grad = True

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )

        # 提取风格向量
        if self.use_aquastyle and self.training:
            style_vec = self.style_extractor(x)
        else:
            style_vec = None

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    x, n=range(self.n_layers), reshape=True, norm=True
                )

                feats_ = []
            
                for i, layer_idx in enumerate(self.extract_ids):
                    feat = feats[layer_idx]  # [B, N, C]
                    # print(feat.shape)
                    B, C, H, W = feat.size()
                    feat_reshaped = feat.view(B, H * W, C)
                    # print(feat_reshaped.shape)

                    # 应用AquaStyle
                    if self.use_aquastyle and style_vec is not None:

                        B, N, C = feat_reshaped.shape
                        dummy_cls = torch.zeros(B, 1, C, device=feat_reshaped.device)
                        feat_with_cls = torch.cat([dummy_cls, feat_reshaped], dim=1)

                        # StyleInjection
                        feat_injected, _ = self.style_injections[i](feat_with_cls, style_vec)
                        feat_injected = feat_injected[:, 1:, :]

                        # Adapter
                        feat_adapter = self.adapters[i](feat_with_cls)
                        feat_adapter = feat_adapter[:, 1:, :]

                        feat = feat_reshaped + feat_injected + feat_adapter

                        B,N,C = feat.size()
                        feat = feat.view(B,C,H,W)
                        # print(feat.shape)
                    feats_.append(feat)

        return feats_'''


class SepAdapterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 64, act=nn.SiLU):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                r, r, kernel_size=3, padding=1, groups=r, bias=False
            ),  # depthwise
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return x


class DenseAdapterLite(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        sizes=(64, 32, 16, 8),
        bottleneck=64,
        share=False,
    ):
        super().__init__()
        self.sizes = list(sizes)

        if share:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck) for _ in self.sizes]
            )
        self.share = share

    def forward(self, feats):
        """
        feats: list of 4 tensors, each [B, C, H_i, W_i]（C = in_dim）
        return: list of 4 tensors, each [B, out_dim, S_i, S_i], S_i ∈ self.sizes
        """
        outs = []
        for i, x in enumerate(feats):
            x = F.interpolate(
                x,
                size=(self.sizes[i], self.sizes[i]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            outs.append(block(x))

        return outs


class LinearAdapter(nn.Module):
    """
    源自 Meta DINOv3 官方评估代码 (linear_head.py) 的极简 Adapter。
    只做通道投影（1x1 Conv）和空间对齐（Bilinear Interpolation）。
    零空间卷积，最大程度保留 DINOv3 原生缺陷特征的锐利度。
    """

    def __init__(
            self,
            in_dim=1024,  # DINOv3 ViT-L 的输出通道数
            out_dim=256,  # 对齐到 fpn_channels * 2
            sizes=(64, 32, 16, 8),  # 你的金字塔特征尺寸
    ):
        super().__init__()
        self.sizes = list(sizes)

        # 官方 Linear Head 做法：BN + 1x1 Conv 降维
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(in_dim),  # 官方习惯在投影前先对原始特征归一化
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
            ) for _ in self.sizes
        ])

    def forward(self, feats):
        """
        feats: DINOV3Wrapper 输出的 4 层特征，形状为 [B, 1024, H_d, W_d]
        return: 尺寸和维度对齐后的 4 层特征，准备进入 PFF
        """
        outs = []
        for i, x in enumerate(feats):
            # 1. 通道降维 1024 -> 256
            x_proj = self.projs[i](x)

            # 2. 空间插值对齐到目标尺寸 (64, 32, 16, 8)
            # 官方评估代码中广泛使用 bilinear 插值
            x_aligned = F.interpolate(
                x_proj,
                size=(self.sizes[i], self.sizes[i]),
                mode="bilinear",
                align_corners=False,
                antialias=True  # DINO特征下采样建议开启抗锯齿
            )
            outs.append(x_aligned)

        return outs

# 第二版adapter
'''class DefectAdapter(nn.Module):
    """轻量版缺陷适配器：极简变换+强保DINOv3缺陷特征+轻量通道强化，完全兼容原接口"""
    def __init__(
        self,
        in_dim=1024,    # DINOv3输入通道（与原一致）
        out_dim=256,    # 输出通道（与原一致，适配FPN）
        sizes=(64, 32, 16, 8),  # 多尺度尺寸（与原一致）
        bottleneck=256, # 轻量瓶颈，减少语义丢失
        share=False,    # 是否共享权重（与原一致）
        defect_gain=0.8 # 轻量缺陷强化，不干扰原始特征
    ):
        super().__init__()
        self.sizes = list(sizes)
        self.share = share
        self.defect_gain = defect_gain

        class LightAdapterBlock(nn.Module):
            def __init__(self, in_dim, out_dim, bottleneck, defect_gain):
                super().__init__()
                self.defect_gain = defect_gain
                # 轻量降维：1×1卷积+BN+GELU（柔和激活，不破坏特征）
                self.reduce = nn.Sequential(
                    nn.Conv2d(in_dim, bottleneck, 1, padding=0, bias=False),
                    nn.BatchNorm2d(bottleneck),
                    nn.GELU()
                )
                # 轻量空间细化：3×1+1×3分组卷积（保留空间关联性）
                self.refine = nn.Sequential(
                    nn.Conv2d(bottleneck, bottleneck, (3,1), padding=(1,0), groups=bottleneck, bias=False),
                    nn.Conv2d(bottleneck, bottleneck, (1,3), padding=(0,1), groups=bottleneck, bias=False),
                    nn.BatchNorm2d(bottleneck),
                    nn.GELU()
                )
                # 轻量缺陷强化：通道注意力（不破坏空间位置）
                self.defect_attn = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(bottleneck, bottleneck//4, 1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(bottleneck//4, bottleneck, 1, bias=False),
                    nn.Sigmoid()
                )
                # 投影升维：与原输出通道一致
                self.project = nn.Sequential(
                    nn.Conv2d(bottleneck, out_dim, 1, padding=0, bias=False),
                    nn.BatchNorm2d(out_dim)
                )
                # 残差连接：与原逻辑一致，强保DINOv3原始特征
                self.residual = nn.Conv2d(in_dim, out_dim, 1, padding=0, bias=False) if in_dim != out_dim else nn.Identity()
                self.final_bn = nn.BatchNorm2d(out_dim)

            def forward(self, x):
                residual = self.residual(x)
                x = self.reduce(x)
                x = self.refine(x)
                # 轻量强化：不干扰缺陷位置
                attn_weight = self.defect_attn(x)
                x = x * (1.0 + self.defect_gain * attn_weight)
                x = self.project(x)
                out = self.final_bn(x + residual)
                return out

        # 模块构建：与原逻辑一致（共享/独立块）
        if self.share:
            self.blocks = nn.ModuleList([LightAdapterBlock(in_dim, out_dim, bottleneck, self.defect_gain)])
        else:
            self.blocks = nn.ModuleList([LightAdapterBlock(in_dim, out_dim, bottleneck, self.defect_gain) for _ in self.sizes])

    def forward(self, feats):
        """前向传播：与原逻辑完全一致，无缝衔接"""


        outs = []
        for i, x in enumerate(feats):
            x_aligned = F.interpolate(
                x,
                size=(self.sizes[i], self.sizes[i]),
                mode="bilinear",
                align_corners=False,
                antialias=True
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            x_adapted = block(x_aligned)
            # print(x_adapted.shape)
            outs.append(x_adapted)
        return outs'''

