import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dinov3.utils.utils import cat_keep_shapes, uncat_with_shapes
from einops import rearrange
from mmcv.ops import MultiScaleDeformableAttention

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


class SpatialPriorCrossAttention(nn.Module):
    def __init__(self, cnn_dim=128, dino_dim=1024, embed_dim=256, num_heads=8, num_points=4):
        super().__init__()

        # 1. 维度映射
        self.q_proj = nn.Conv2d(cnn_dim, embed_dim, kernel_size=1)
        # 传统注意力的 K 在这里被省略了，Deformable Attn 直接从 Query 预测偏移量，只对 V 采样
        self.v_proj = nn.Conv2d(dino_dim, embed_dim, kernel_size=1)

        # 2. 引入 MMCV 的可变形注意力核心算子
        # 此时只处理单层尺度的跨模态对齐，因此 num_levels=1
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dim,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            batch_first=True
        )

        # 3. 输出特征的平滑与对齐
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def get_reference_points(self, H_q, W_q, device):
        """
        生成查询向量 (Query) 的归一化参考点坐标。
        它指示了 CNN 特征图上的每个像素，应该去 DINO 特征图的大致对应位置开始搜寻。
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_q - 0.5, H_q, dtype=torch.float32, device=device) / H_q,
            torch.linspace(0.5, W_q - 0.5, W_q, dtype=torch.float32, device=device) / W_q,
            indexing='ij'
        )
        # 变形要求: [Batch(1), H*W, Levels(1), 2(x, y)]
        ref = torch.stack((ref_x, ref_y), dim=-1).reshape(1, H_q * W_q, 1, 2)
        return ref

    def forward(self, cnn_feat, dino_feat):
        B, _, H_c, W_c = cnn_feat.shape
        _, _, H_d, W_d = dino_feat.shape

        # 1. 生成 Query 和 Value
        Q = self.q_proj(cnn_feat)
        V = self.v_proj(dino_feat)

        # 2. 形状展平适应 Transformer 输入: [B, C, H, W] -> [B, H*W, C]
        Q_flat = rearrange(Q, 'b c h w -> b (h w) c')
        V_flat = rearrange(V, 'b c h w -> b (h w) c')

        # 3. 构造 Deformable Attention 需要的元数据
        # 参考点：复制到对应的 Batch Size
        reference_points = self.get_reference_points(H_c, W_c, Q.device).repeat(B, 1, 1, 1)

        # Value 的空间形状和级别的起始索引
        spatial_shapes = torch.as_tensor([[H_d, W_d]], dtype=torch.long, device=Q.device)
        level_start_index = torch.zeros((1,), dtype=torch.long, device=Q.device)

        # 4. 执行可变形交叉注意力计算
        attn_out = self.deform_attn(
            query=Q_flat,
            key=None,
            value=V_flat,
            identity=None,
            query_pos=None,
            key_padding_mask=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        # 5. 还原空间维度并执行残差连接
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=H_c, w=W_c)
        out = self.out_proj(attn_out)

        # CNN 特征残差融入网络
        return out + Q

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


class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        extract_ids=list(range(24)),
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

        apply_bottleneck_adapter_to_dinov3(
            self.model,
            target_layers=target_layers,
            dim=1024,
            bottleneck_dim=64,  # 你可以根据显存调整，通常 64 或 128
            use_attn=False,
            use_mlp=True
        )

        for name, p in self.model.named_parameters():
            if "adapter" in name:
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

        return feats

class ViTAdapterLike(nn.Module):
    def __init__(self, cnn_dim=128, dino_dim=1024, out_dim=256, num_scales=4):
        super().__init__()
        # 为4个不同的尺度分别实例化交叉注意力模块
        self.cross_attns = nn.ModuleList([
            SpatialPriorCrossAttention(
                cnn_dim=cnn_dim,
                dino_dim=dino_dim,
                embed_dim=out_dim
            ) for _ in range(num_scales)
        ])

    def forward(self, cnn_feats, dino_feats):
        """
        cnn_feats: list of 4 tensors from ResNet [B, 128, H_i, W_i]
        dino_feats: list of 4 tensors from DINO GroupWeightFusion [B, 1024, H_d, W_d]
        """
        outs = []
        for i in range(len(cnn_feats)):
            # 用 cnn_feats[i] 的空间分辨率，去重塑 dino_feats[i] 的信息
            adapted_feat = self.cross_attns[i](cnn_feats[i], dino_feats[i])
            outs.append(adapted_feat)
        return outs

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
