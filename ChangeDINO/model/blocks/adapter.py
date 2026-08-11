import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dinov3.utils.utils import cat_keep_shapes, uncat_with_shapes
from einops import rearrange
from mmcv.ops import MultiScaleDeformableAttention
import math


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

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.ops import MultiScaleDeformableAttention

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


class ParallelMLPAdapterWrapper(nn.Module):
    """将 Adapter 与 MLP 并行"""

    def __init__(self, mlp_module, in_dim, bottleneck_dim):
        super().__init__()
        self.mlp_module = mlp_module
        self.adapter = BottleneckAdapter(in_dim, bottleneck_dim)

    def forward(self, x):
        # MLP 和 Adapter 并行处理相同的输入 x
        mlp_out = self.mlp_module(x)
        adp_out = self.adapter(x)
        return mlp_out + adp_out

    def forward_list(self, x_list):
        """兼容 DINOv3 内部特有的列表前向传播机制"""
        mlp_out_list = self.mlp_module.forward_list(x_list)

        # 注意：这里传入的是 x_list，而不是 mlp_out_list
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
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
                # blocks[i].mlp = SequentialMLPAdapterWrapper(blocks[i].mlp, dim, bottleneck_dim)
                blocks[i].mlp = ParallelMLPAdapterWrapper(blocks[i].mlp, dim, bottleneck_dim)
    return dinov3_model


class LoRALinear(nn.Module):
    """论文式 LoRA：冻结原 Linear，旁路加 (alpha/r) * B(A(x))，B 初始化为 0"""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.in_features = in_f
        self.out_features = out_f
        self.bias = base.bias
        self.r = r
        self.alpha = alpha
        self.scale = alpha / max(1, r)                 # 16/8 = 2
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Linear(in_f, r, bias=False)
            self.lora_B = nn.Linear(r, out_f, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)          # 起步等于原模型
        else:
            self.lora_A = None
            self.lora_B = None
        for p in self.base.parameters():
            p.requires_grad = False                     # base 冻结
        try:
            dev = next(self.base.parameters()).device
            self.to(dev)
        except StopIteration:
            pass

    def forward(self, x):
        out = self.base(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            if self.lora_A.weight.device != x.device:
                self.lora_A.to(x.device)
                self.lora_B.to(x.device)
            out = out + self.scale * self.lora_B(self.dropout(self.lora_A(x)))
        return out


def apply_lora_to_dinov3(model, target_layers=None, rank=8, alpha=16.0, dropout=0.0):
    """给 DINOv3 每个 block 的 attn.qkv/proj 和 mlp.fc1/fc2 挂 LoRA"""
    if target_layers is None:
        target_layers = list(range(len(model.blocks)))
    for i in target_layers:
        if i >= len(model.blocks):
            continue
        blk = model.blocks[i]
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r=rank, alpha=alpha, dropout=dropout)
        blk.attn.proj = LoRALinear(blk.attn.proj, r=rank, alpha=alpha, dropout=dropout)
        blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r=rank, alpha=alpha, dropout=dropout)
        blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r=rank, alpha=alpha, dropout=dropout)
    return model


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

        # 论文式 LoRA：24 个 block 全部注入（qkv/proj/fc1/fc2），r=8, alpha=16
        apply_lora_to_dinov3(
            self.model,
            target_layers=list(range(self.n_layers)),  # 0..23 全部
            rank=8,
            alpha=16.0,
            dropout=0.0,
        )

        for name, p in self.model.named_parameters():
            if "lora" in name:
                p.requires_grad = True

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            feats = self.model.get_intermediate_layers(
                x, n=range(self.n_layers), reshape=True, norm=True
            )
            feats_ = []
            for i in range(len(self.extract_ids)):
                feats_.append(feats[self.extract_ids[i]].float())  # 关键：转回 fp32

        return feats_



class LinearAdapter(nn.Module):

    def __init__(
            self,
            in_dim=1024,
            out_dim=256,
            sizes=(128, 64, 32, 16),
    ):
        super().__init__()
        self.sizes = list(sizes)

        # 官方 Linear Head 做法：BN + 1x1 Conv 降维
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
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

            x_proj = self.projs[i](x)

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

class ConvOut(nn.Module):
    def __init__(self, in_channel):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.conv(x)
