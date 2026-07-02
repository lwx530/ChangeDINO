import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dinov3.utils.utils import cat_keep_shapes, uncat_with_shapes

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


class BottleneckAdapter(nn.Module):
    """传统的 降维-激活-升维 Adapter"""

    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, in_dim)

        # 核心：将 up_proj 初始化为 0，使得训练初期 Adapter 像一个 Identity 映射，不破坏预训练权重
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.up_proj(self.act(self.down_proj(x)))


class ParallelAttnAdapterWrapper(nn.Module):
    """将 Adapter 与原 Self-Attention 并行"""

    def __init__(self, attn_module, in_dim, bottleneck_dim):
        super().__init__()
        self.attn_module = attn_module
        self.adapter = BottleneckAdapter(in_dim, bottleneck_dim)

    def forward(self, x, attn_bias=None, rope=None):
        # 原 Attention 分支
        attn_out = self.attn_module(x, attn_bias=attn_bias, rope=rope)
        # Adapter 并行分支
        adp_out = self.adapter(x)
        return attn_out + adp_out

    def forward_list(self, x_list, attn_bias=None, rope_list=None):
        """兼容 DINOv3 内部特有的列表前向传播机制"""
        attn_out_list = self.attn_module.forward_list(x_list, attn_bias=attn_bias, rope_list=rope_list)

        # 将 list 拼在一起过 Adapter 可以加速计算
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        adp_flat = self.adapter(x_flat)
        adp_out_list = uncat_with_shapes(adp_flat, shapes, num_tokens)

        return [a + adp for a, adp in zip(attn_out_list, adp_out_list)]


class SequentialMLPAdapterWrapper(nn.Module):
    """将 Adapter 串联在 MLP 之后"""

    def __init__(self, mlp_module, in_dim, bottleneck_dim):
        super().__init__()
        self.mlp_module = mlp_module
        self.adapter = BottleneckAdapter(in_dim, bottleneck_dim)

    def forward(self, x):
        mlp_out = self.mlp_module(x)
        # 串联：Adapter 的输入是 MLP 的输出
        adp_out = self.adapter(mlp_out)
        return mlp_out + adp_out

    def forward_list(self, x_list):
        mlp_out_list = self.mlp_module.forward_list(x_list)

        x_flat, shapes, num_tokens = cat_keep_shapes(mlp_out_list)
        adp_flat = self.adapter(x_flat)
        adp_out_list = uncat_with_shapes(adp_flat, shapes, num_tokens)

        return [m + adp for m, adp in zip(mlp_out_list, adp_out_list)]


def apply_bottleneck_adapter_to_dinov3(dinov3_model, target_layers, dim=1024, bottleneck_dim=64, use_attn=True,
                                       use_mlp=False):
    """注入函数"""
    blocks = dinov3_model.blocks
    for i in target_layers:
        if i < len(blocks):
            if use_attn:
                blocks[i].attn = ParallelAttnAdapterWrapper(blocks[i].attn, dim, bottleneck_dim)
            if use_mlp:
                blocks[i].mlp = SequentialMLPAdapterWrapper(blocks[i].mlp, dim, bottleneck_dim)
    return dinov3_model

def sal_guide(feature, sal, number):
    feature_list = torch.chunk(feature, number, dim=1)
    out = torch.cat((feature_list[0], sal), dim=1)
    for i in range(1, number):
        out = torch.cat((out, feature_list[i], sal), dim=1)
    return out

class PreBlockAdapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x, rope=None):
        prompt = self.prompt_learn(x)
        return self.block(x + prompt, rope)

    def forward_list(self, x_list, rope_list=None):
        return [self.block(x + self.mlp(x), rope)
                for x, rope in zip(x_list, rope_list if rope_list else [None] * len(x_list))]

class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
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

        target_layers = [6, 8, 10, 12, 14, 16, 18, 20]

        blocks = self.model.blocks
        for i in target_layers:
            blocks[i] = PreBlockAdapter(blocks[i])

        for name, p in self.model.named_parameters():
            if "prompt_learn" in name:
                p.requires_grad = True

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )

        with torch.autocast(device_type=self.device, dtype=torch.float32):
            feats = self.model.get_intermediate_layers(
                x, n=range(self.n_layers), reshape=True, norm=True
            )
            feats_ = []
            for i in range(len(self.extract_ids)):
                feats_.append(feats[self.extract_ids[i]])  # [B, N, C]
        return feats_

class LinearAdapter(nn.Module):

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
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            ) for _ in self.sizes
        ])
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):

        outs = []
        for i, x in enumerate(feats):
            # 1. 通道降维 1024 -> 256
            x_proj = self.projs[i](x)

            # 2. 空间插值对齐到目标尺寸 (64, 32, 16, 8)
            x_aligned = F.interpolate(
                x_proj,
                size=(self.sizes[i], self.sizes[i]),
                mode="bilinear",
                align_corners=False,
                antialias=True  # DINO特征下采样建议开启抗锯齿
            )
            x2 = self.conv2(x_aligned)
            outs.append(x2)

        return outs


