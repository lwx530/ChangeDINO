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


class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=1.0):
        super().__init__()
        self.base = base_linear

        # 保持原始属性，防止 DINOv3 内部调用报错
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # 确保原始全连接层被彻底冻结
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha

        # LoRA 的 A 和 B 矩阵
        self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        # 论文标准做法：残差相加
        return base_out + self.alpha * lora_out


def apply_lora_to_dinov3_official(dinov3_model, r=8, alpha=1.0, verbose=False):
    """
    针对 torch.hub 加载的 Meta 官方 DINOv3 的 LoRA 注入函数
    """
    blocks = dinov3_model.blocks  # 官方 DINOv3 的 Transformer Blocks

    for i, block in enumerate(blocks):
        if verbose:
            print(f"[LoRA] 正在将 LoRA 注入到 Block {i} 的 qkv 和 proj 层")

        # 将标准的 Linear 替换为带有旁路矩阵的 LoRALinear
        block.attn.qkv = LoRALinear(block.attn.qkv, r=r, alpha=alpha)
        block.attn.proj = LoRALinear(block.attn.proj, r=r, alpha=alpha)

    return dinov3_model

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