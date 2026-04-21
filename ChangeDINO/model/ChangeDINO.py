import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import matplotlib.pyplot as plt
import numpy as np

from .blocks.fpn import FPN, DsBnRelu
from .blocks.cbam import CBAM
from .blocks.adapter import DINOV3Wrapper, DenseAdapterLite, LinearAdapter
from .blocks.diffatts import TransformerBlock
from .blocks.refine import LearnableSoftMorph
from .blocks.sfhm import SFHM
from .backbone.mobilenetv2 import mobilenet_v2


class InNetworkAdapterWrapper(nn.Module):
    def __init__(self, original_block, dim=1024, bottleneck_dim=256):
        super().__init__()
        # 1. 挂载原始的 DINOv3 Block (冻结状态)
        self.original_block = original_block

        # 2. 瓶颈结构的线性 Adapter (1024 -> 256 -> 1024)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)

        # 3. 核心：零初始化，确保初始状态等价于原生 DINOv3
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, *args, **kwargs):
        # 第一步：数据正常穿过原始的 DINOv3 层。
        # 必须带上 *args, **kwargs，因为 DINOv3 会传入 rope_sincos 等位置编码参数
        out = self.original_block(x, *args, **kwargs)

        # 第二步：将输出特征送入 Adapter 进行特异性加工
        adapter_out = self.up(self.act(self.down(out)))

        # 第三步：残差融合并返回。
        # 这个返回值会顺着 DINOv3 的源码，作为输入直接流向下一层 (比如从第5层流向第6层)！
        return out + adapter_out

class SRFMaskGenerator(nn.Module):
    """
    SRF (Spatial Refinement) 掩码生成器
    作用：从最浅层、物理边界最清晰的 CNN 特征中，提取出一个 0~1 的高清边界权重图。
    """
    def __init__(self, in_channels=128):
        super().__init__()
        # 使用轻量级卷积将 128 维压缩到 1 维
        self.mask_gen = nn.Sequential(
            # 第一层：降维并提取关键边界
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 第二层：压缩为单通道灰度掩码
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            # 关键：Sigmoid 确保输出的乘法权重在 0 到 1 之间
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mask_gen(x)

def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class PyramidFeatureFusion(nn.Module):
    def __init__(
            self,
            in_dims=[128, 128, 128, 128],
            dense_dim=1024,
            patch_size=16,
            hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        '''# 定义一个简单的纯卷积融合块
        def make_fusion_block(in_channels, out_channels):
            return nn.Sequential(
                # 第一步：你的原有的深度可分离卷积/特征降维块
                DsBnRelu(in_channels, out_channels),
                # 第二步：纯卷积块（替代了原来的 CBAM）
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.c4 = make_fusion_block(in_dims[3] + hidden_dim, in_dims[3])
        self.c3 = make_fusion_block(in_dims[2] + hidden_dim, in_dims[2])
        self.c2 = make_fusion_block(in_dims[1] + hidden_dim, in_dims[1])
        self.c1 = make_fusion_block(in_dims[0] + hidden_dim, in_dims[0])'''

        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]), CBAM(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]), CBAM(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]), CBAM(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]), CBAM(in_dims[0], 8)
        )

    def forward(self, feas, ds_fea):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_fea # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4


class Encoder(nn.Module):
    def __init__(
            self,
            backbone="mobilenetv2",
            fpn_channels=128,
            deform_groups=4,
            gamma_mode="SE",
            beta_mode="contextgatedconv",
            # dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            dino_weight="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            device="cuda",
            # extract_ids=[5, 11, 17, 23],
            extract_ids=list(range(24)),
            **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(weights_path=dino_weight, device=device, extract_ids=extract_ids)

        self.dense_adp = DenseAdapterLite(
            in_dim=1024, out_dim=dense_out_dim, bottleneck=fpn_channels // 2,
        )

        self.defect_adapter = LinearAdapter(
            in_dim=1024,
            out_dim=dense_out_dim,  # 即 256
            sizes=(64, 32, 16, 8)
        )

        '''# ==================== 新增：内嵌式 Adapter 动态注入 ====================
        # 目标层级：在第 5, 11, 17, 23 层的 Block 外面套上我们的 Wrapper
        target_layers = [5, 11, 17, 23]

        # ⚠️ 注意：这里的 `self.dino` 是你在 adapter.py 写的 DINOV3Wrapper 类。
        # 你需要根据 DINOV3Wrapper 内部定义真实 DINO 模型的变量名，来找到 `blocks`。
        # 假设真实模型在 wrapper 里叫做 `self.model`，那么路径就是 `self.dino.model.blocks`。
        # 如果报错找不到 blocks，请查看 adapter.py 里你的实例化名称（也可能是 self.dino.dino.blocks 等）。

        for idx in target_layers:
            original_block = self.dino.model.blocks[idx]  # 提取原 Block
            self.dino.model.blocks[idx] = InNetworkAdapterWrapper(
                original_block, dim=1024, bottleneck_dim=256
            )  # 偷梁换柱
        # ===================================================================='''

        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

        # 【新增】实例化 SRF 掩码生成器
        # 我们将接收 FPN 输出的最高分辨率层 (通道数为 fpn_channels)
        # =========================================================
        self.srf_mask_gen = SRFMaskGenerator(in_channels=fpn_channels)

        # ==================== 新增：SFHM 前置处理模块 ====================
        # 因为我们要把 FPN(128) 和 Adapter(256) 拼起来，维度会变成 384
        # 需要先用一个 1x1 卷积降维回 128，再送入 SFHM
        self.fusion_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels + dense_out_dim, fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # 实例化 4 个尺度的 SFHM 模块
        self.sfhm_modules = nn.ModuleList([
            SFHM(in_dim=fpn_channels) for _ in range(4)
        ])
        # ===============================================================

        dino_adapted_ch = fpn_channels * 2

        self.dino_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dino_adapted_ch, 1, kernel_size=1, bias=True),
                nn.Sigmoid()  # 压缩到 0~1 之间，作为概率权重
            ) for _ in range(4)
        ])

    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """

        fea = self.backbone.forward(x)
        fea = self.fpn(fea[-4:])  # t1_p1, t1_p2, t1_p3, t1_p4

        # 【SRF 关键点 1】：提取“铅笔线稿”
        # fea[0] 是分辨率最高(例如 64x64)、未经深层语义污染的物理细节特征
        # =========================================================
        highest_res_detail = fea[0]

        # ds_fea = self.dino(x)  # [B, N, C]

        raw_ds_fea = self.dino(x)  # 获取24层

        # ==================== 更新：24层全特征分组聚合逻辑 ====================
        # 将 24 层特征等分为 4 组，每组 6 层。
        # 采用求平均 (mean) 的方式，保证特征的量级稳定
        ds_fea = []
        for i in range(4):
            # 取出当前层的 6 个特征图
            group_feats = raw_ds_fea[i * 6: (i + 1) * 6]

            # 将 6 个特征图在新的维度(dim=0)堆叠起来，然后求平均
            # shape 变化: 6 个 [B, N, C] -> [6, B, N, C] -> mean -> [B, N, C]
            group_mean_feat = torch.mean(torch.stack(group_feats, dim=0), dim=0)

            ds_fea.append(group_mean_feat)
        # ==========================================================

        # process dense features
        ds_fea_adapted = self.defect_adapter(ds_fea)

        # ==================== 新增：空频混合增强逻辑 ====================
        enhanced_feas = []

        for i in range(4):
            # 将对应尺度的 CNN 特征和 DINO 特征拼接: 128 + 256 = 384
            # fused = torch.cat([fea[i], ds_fea_adapted[i]], dim=1)

            # 降维到 128
            # fused = self.fusion_projs[i](fused)

            # ==========================================
            # 核心：通过 SFHM 模块进行 2D-FFT 频域强化和空域提纯
            # 这里会自动放大高频缺陷（划痕/裂纹），抑制低频背景
            # ==========================================
            sfhm_out = self.sfhm_modules[i](fea[i])
            # 第二步：DINO 生成空间注意力门控图
            # 告诉网络：哪些地方是真正的异常，哪些地方是安全背景
            gate = self.dino_gates[i](ds_fea_adapted[i])

            # 第三步：显式相乘！(广播机制)
            # 背景区：高频噪声 * 0(gate) ≈ 0 （噪声被完美抹除）
            # 缺陷区：高频划痕 * 1(gate) ≈ 高频划痕 （缺陷被完美保留）
            gated_sfhm_out = sfhm_out * gate
            enhanced_feas.append(gated_sfhm_out)
            # 为了兼容你原有的 PFF (它期望收到两组特征去运算)
            # 我们直接把 SFHM 提纯后的特征映射回 256 维当作纯净的 ds_fea_adapted
            # 或者最简单的做法是：不用 PFF 里的 ds_fea_adapted 逻辑了，但为了少改代码，保持双路输入
        # ===============================================================

        # 4. 把经过 SFHM 极致强化的特征，送入你原有的 PFF 做最终金字塔融合
        # 注意：因为你的 PFF 是把 fea 和 ds_fea 再次拼接处理，
        # 为了不破坏你的原网络拓扑，fea 传入增强后的，ds_fea 依然传入原本的。
        # 这样 enhanced_feas 里的高频极度锐利，PFF 融合时会更依赖这些高频信息。
        final_fea = self.pff(enhanced_feas, ds_fea_adapted)

        '''# 将 PFF 的输出解包为四个尺度的特征图
        x1, x2, x3, x4 = final_fea
        # fea = self.pff(fea, ds_fea_adapted)
        # 【SRF 关键点 2】：执行边界锐化
        # =========================================================
        # A. 用浅层特征生成高清边界掩码 (Shape: [B, 1, H, W])
        edge_mask = self.srf_mask_gen(highest_res_detail)
        # B. 残差相乘锐化。
        # 原理：在边缘掩码强烈(趋近于1)的地方，深层特征 x1 的激活值翻倍；
        # 在没有物理边缘的平坦区(趋近于0)，x1 保持原样 (x1 * 1.0)。
        x1_sharpened = x1 * (1.0 + edge_mask)'''

        # return x1_sharpened, x2, x3, x4
        return final_fea


class FuseGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)


class Detector(nn.Module):
    def __init__(
            self,
            fpn_channels=128,
            n_layers=[1, 1, 1, 1],
            num_classes=2,  # 新增：分割类别数，默认2类（缺陷/背景）
            **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes

        # 保持门控融合模块
        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

        # 修改Transformer注意力机制：CDA → 普通自注意力
        self.tb5 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="CDA",
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=3,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree", )
                for _ in range(n_layers[0])]
        )
        self.tb4 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="CDA",
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=3,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree", )
                for _ in range(n_layers[1])]
        )
        self.tb3 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="OCDA",
                window_size=8,
                overlap_ratio=0.5,
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=2,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            )
                for _ in range(n_layers[2])]
        )
        self.tb2 = nn.Sequential(
            *[TransformerBlock(
                dim=fpn_channels,
                spatial_attn_type="OCDA",
                window_size=8,
                overlap_ratio=0.5,
                num_channel_heads=8,
                num_spatial_heads=4,
                depth=1,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="BiasFree",
            )
                for _ in range(n_layers[3])]
        )
        self.p5_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p4_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p3_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p2_head = nn.Conv2d(fpn_channels, num_classes, 1)

    def forward(self, xs):
        ### Extract backbone features
        '''t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s'''

        # 1. 解包特征金字塔（单输入，没有差异计算
        fea_p2, fea_p3, fea_p4, fea_p5 = xs

        # 2. 自顶向下处理（与原来类似，但没有diff计算）
        # 从最深层的p5开始
        # fea_p5 = self.tb5(fea_p5)
        pred_p5 = self.p5_head(fea_p5)

        # p5特征融合到p4
        fea_p4 = self.p5_to_p4(fea_p5, fea_p4)
        # fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)

        # p4特征融合到p3
        fea_p3 = self.p4_to_p3(fea_p4, fea_p3)
        # fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)

        # p3特征融合到p2
        fea_p2 = self.p3_to_p2(fea_p3, fea_p2)
        # fea_p2 = self.tb2(fea_p2)
        pred_p2 = self.p2_head(fea_p2)

        # 3. 上采样到统一尺寸
        pred_p2 = F.interpolate(
            pred_p2, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=(256, 256), mode="bilinear", align_corners=False
        )
        pred_p5 = F.interpolate(
            pred_p5, size=(256, 256), mode="bilinear", align_corners=False
        )

        return pred_p2, pred_p3, pred_p4, pred_p5


class ChangeModel(nn.Module):
    def __init__(self, backbone="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1, 1], **kwargs):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.detector = Detector(fpn_channels=fpn_channels, n_layers=n_layers, **kwargs)
        self.refiner = LearnableSoftMorph(3, 5)

    @torch.inference_mode()
    def _forward(self, x):
        # for inference
        fea = self.encoder(x)
        # fea, dino_feats, adapter_feats = self.encoder(x)      # 将DINOv3提取的特征图和经过adapter处理的特征图可视化
        pred, _, _, _ = self.detector(fea)
        pred = self.refiner(pred)
        # return pred, dino_feats, adapter_feats      # 这个在测试时要用到可视化的时候用
        return pred                   # 训练的时候用这个

    def forward(self, x):
        # for training
        ## change detection
        fea = self.encoder(x)
        # fea, dino_feats, adapter_feats = self.encoder(x)      # 将DINOv3提取的特征图和经过adapter处理的特征图可视化
        # fea2 = self.encoder(x2)

        preds = self.detector(fea)
        final_pred = self.refiner(preds[0])
        return final_pred, preds  # pred, pred_p2, pred_p3, pred_p4, pred_p5
