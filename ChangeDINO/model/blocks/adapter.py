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

        # ==================== 新增：注入 Adapter ====================
        # DINOv3-Large 的维度是 1024
        target_layers = [3, 5, 9, 11, 15, 17, 21, 23]

        apply_bottleneck_adapter_to_dinov3(
            self.model,
            target_layers=target_layers,
            dim=1024,
            bottleneck_dim=64,  # 你可以根据显存调整，通常 64 或 128
            use_attn=True,  # Attention 旁的并行 Adapter
            use_mlp=True  # MLP 后的串联 Adapter (解答你的第二个问题)
        )

        # 2. 重新解冻刚才注入的 adapter 的参数，确保能够反向传播
        for name, p in self.model.named_parameters():
            if "adapter" in name:
                p.requires_grad = True
        # =========================================================

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )
        # 因为 Adapter 需要计算梯度，如果你用了 no_grad()，Adapter 就无法训练了！
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            feats = self.model.get_intermediate_layers(
                x, n=range(self.n_layers), reshape=True, norm=True
            )
            feats_ = []
            for i in range(len(self.extract_ids)):
                feats_.append(feats[self.extract_ids[i]])  # [B, N, C]
        return feats_


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
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
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


class FidelityAwareAdapter(nn.Module):
    """
    基于 Dino U-Net FAPM 模块思想的高保真适配器 (Fidelity-Aware Adapter)
    核心机制: 提取共享的低秩上下文，生成空间级的 affine 参数 (gamma, beta)，
             对各个尺度的特征进行自适应调制，防止 1024 维降维时的细节丢失。
    """

    def __init__(
            self,
            in_dim=1024,
            out_dim=256,
            sizes=(64, 32, 16, 8),
            rank=256,  # 低秩共享空间的维度
    ):
        super().__init__()
        self.sizes = list(sizes)
        self.rank = rank

        # 1. 共享上下文分支 (Shared Context Conv)
        self.shared_conv = nn.Conv2d(in_dim, rank, kernel_size=1, bias=False)

        self.specific_convs = nn.ModuleList()
        self.modulators = nn.ModuleList()
        self.refine_convs = nn.ModuleList()

        for _ in self.sizes:
            # 2. 尺度特定分支 (Scale-Specific Conv)
            self.specific_convs.append(
                nn.Conv2d(in_dim, rank, kernel_size=1, bias=False)
            )

            # 3. 调制参数生成器 (Modulator Generator)
            # 接收 rank 维的 shared context，输出 rank*2 维的参数 (gamma 和 beta)
            self.modulators.append(nn.Sequential(
                nn.Conv2d(rank, rank, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(rank, rank * 2, kernel_size=1)
            ))

            # 4. 提纯与维度映射 (Refinement Stage)
            # 采用深度可分离卷积捕获局部细节，并映射到目标输出维度
            self.refine_convs.append(nn.Sequential(
                nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False),  # Depthwise
                nn.BatchNorm2d(rank),
                nn.GELU(),
                nn.Conv2d(rank, out_dim, kernel_size=1, bias=True)  # Pointwise
            ))

    def forward(self, feats):
        """
        feats: DINOV3Wrapper 输出的 4 层特征 (通常分辨率相同，皆为 1024 维)
        """
        outs = []
        for i, x in enumerate(feats):
            # A. 正交分解：提取共享上下文 (Z_ctx) 和 尺度特定特征 (Z_sp)
            z_ctx = self.shared_conv(x)  # [B, rank, H, W]
            z_sp = self.specific_convs[i](x)  # [B, rank, H, W]

            # B. 上下文引导调制 (Feature-wise Modulation)
            # 使用 z_ctx 生成空间感知的 gamma 和 beta
            params = self.modulators[i](z_ctx)  # [B, rank*2, H, W]
            gamma, beta = torch.chunk(params, 2, dim=1)  # 各自 [B, rank, H, W]

            # 执行调制: Z_mod = Z_sp * (1 + gamma) + beta
            z_mod = z_sp * (1.0 + gamma) + beta

            # C. 空间对齐 (Interpolation)
            # 先对齐到目标尺寸 (64, 32, 16, 8)
            z_mod_aligned = F.interpolate(
                z_mod,
                size=(self.sizes[i], self.sizes[i]),
                mode="bilinear",
                align_corners=False,
                antialias=True
            )

            # D. 局部提纯与输出 (Refinement)
            out = self.refine_convs[i](z_mod_aligned)
            outs.append(out)

        return outs