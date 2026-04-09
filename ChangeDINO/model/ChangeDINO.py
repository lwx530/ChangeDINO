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


'''class DiscreteWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # 这里的实现模拟了 Haar 小波变换
        # 将图像 reshape 成 [B, C, H/2, 2, W/2, 2]
        x_reshaped = x.view(b, c, h // 2, 2, w // 2, 2)

        # 提取四个分量: x00(左上), x01(右上), x10(左下), x11(右下)
        x00 = x_reshaped[:, :, :, 0, :, 0]  # Even rows, Even cols
        x01 = x_reshaped[:, :, :, 0, :, 1]  # Even rows, Odd cols
        x10 = x_reshaped[:, :, :, 1, :, 0]  # Odd rows, Even cols
        x11 = x_reshaped[:, :, :, 1, :, 1]  # Odd rows, Odd cols

        # Haar Wavelet 公式
        # LL: Low frequency (Approximation)
        LL = x00 + x01 + x10 + x11
        # LH: Horizontal High freq (Detail)
        LH = x00 + x01 - x10 - x11
        # HL: Vertical High freq (Detail)
        HL = x00 - x01 + x10 - x11
        # HH: Diagonal High freq (Detail)
        HH = x00 - x01 - x10 + x11

        # 为了数值稳定性，通常除以2 (有些实现除以4，这里保持量级一致即可)
        return LL / 2, LH / 2, HL / 2, HH / 2


class SemanticFrequencyDifferential(nn.Module):
    """
    改进方案3的核心模块：语义引导的频域差分
    输入: Backbone的高频细节 + DINOv3的语义上下文
    输出: 经过语义过滤的缺陷特征 (模拟 Difference Map)
    """

    def __init__(self, backbone_dim, dino_dim, out_dim):
        super().__init__()
        self.dwt = DiscreteWaveletTransform()

        # 1. 高频特征融合层
        # DWT产生3个高频分量(LH, HL, HH)，通道数变为 backbone_dim * 3
        self.high_freq_fusion = nn.Sequential(
            nn.Conv2d(backbone_dim * 3, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 语义门控生成器 (Semantic Gating)
        # 将 DINO 特征转化为 0~1 的权重图
        self.gate_generator = nn.Sequential(
            nn.Conv2d(dino_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            nn.Sigmoid()  # 生成门控权重
        )

        # 3. 最终融合 (可选：是否把低频信息也加回去？)
        # 方案3重点是高频，但为了恢复结构，通常保留一部分原始特征
        self.final_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, f_backbone, f_dino):
        # f_backbone: [B, 128, H, W] (来自FPN)
        # f_dino: [B, 256, H', W'] (来自Adapter)

        # Step 1: 小波分解
        # output size: H/2, W/2
        LL, LH, HL, HH = self.dwt(f_backbone)

        # Step 2: 聚合高频信息
        high_freq = torch.cat([LH, HL, HH], dim=1)  # [B, 128*3, H/2, W/2]
        high_freq_feat = self.high_freq_fusion(high_freq)  # [B, 128, H/2, W/2]

        # Step 3: 处理语义特征生成门控
        # DINO特征尺寸可能与DWT后的尺寸不一致，需要插值对齐
        f_dino_resized = F.interpolate(
            f_dino,
            size=high_freq_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        gate = self.gate_generator(f_dino_resized)

        # Step 4: 核心逻辑 - 语义抑制噪音
        # Gate值越接近1，表示该高频大概率是缺陷；接近0表示是背景纹理
        # 注意：这里我们假设DINO能识别"背景"，所以我们想要的是 "高频" AND "非背景"
        # 或者让网络自己学习Gate：Gate高响应区域 = 缺陷区域
        diff_map = high_freq_feat * gate

        # Step 5: 恢复尺寸 (为了送入Detector)
        # 上采样回 H, W
        diff_map_up = F.interpolate(
            diff_map,
            size=f_backbone.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 可选：加上原始FPN特征的残差，防止丢失过多结构信息
        # out = self.final_proj(diff_map_up + f_backbone)
        # 这里为了强调"差分"概念，我们直接返回处理后的高频差分图
        out = self.final_proj(diff_map_up)

        return out'''


def save_feature_map(feature_tensor, save_name, save_dir="vis_results-3"):
    """
    将 [B, C, H, W] 的特征图压缩并保存为热力图
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 1. 取 Batch 中的第一张图: 变成 [C, H, W]
    if feature_tensor.dim() == 4:
        feat = feature_tensor[0]
    else:
        feat = feature_tensor

    # 2. 沿通道维度取平均，得到空间响应强度: [H, W]
    # 使用 .detach().cpu().numpy() 将其转为 numpy 数组以便画图
    feat_map = torch.mean(feat, dim=0).detach().cpu().numpy()

    # 可选：对特征图进行归一化，让图像对比度更好
    feat_map = (feat_map - np.min(feat_map)) / (np.max(feat_map) - np.min(feat_map) + 1e-8)

    # 3. 绘制并保存
    plt.figure(figsize=(6, 6))
    plt.imshow(feat_map, cmap='jet')  # jet 伪彩色：红色代表高响应，蓝色代表低响应
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title(save_name)
    plt.savefig(os.path.join(save_dir, f"{save_name}.png"), bbox_inches='tight')
    plt.close()

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

        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

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

        '''# 4. [新增] 语义频域差分模块 (SFD)
        # 我们有4个层级的特征，所以需要4个SFD模块
        # 输入: FPN特征(128) + Adapter特征(256) -> 输出(128)
        self.sfd_modules = nn.ModuleList([
            SemanticFrequencyDifferential(
                backbone_dim=fpn_channels,  # 128
                dino_dim=dense_out_dim,  # 256
                out_dim=fpn_channels  # 128 (必须匹配Detector的输入维度)
            ) for _ in range(4)
        ])'''

    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """

        # ==================== 可视化控制开关 ====================
        # 建议：仅在 batch_size=1 且你想看图的时候设为 True，平时训练设为 False
        VISUALIZE = False
        # ========================================================

        fea = self.backbone.forward(x)
        fea = self.fpn(fea[-4:])  # t1_p1, t1_p2, t1_p3, t1_p4


        # ds_fea = self.dino(x)  # [B, N, C]

        # 2. DINOv3 支路获取 8 层语义特征
        raw_ds_fea = self.dino(x)  # 这里 raw_ds_fea 是一个包含 8 个张量的 list

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
        # =====================================================================

        # process dense features
        ds_fea_adapted = self.defect_adapter(ds_fea)

        # ==================== 新增：空频混合增强逻辑 ====================
        enhanced_feas = []

        for i in range(4):
            # 将对应尺度的 CNN 特征和 DINO 特征拼接: 128 + 256 = 384
            fused = torch.cat([fea[i], ds_fea_adapted[i]], dim=1)

            # 降维到 128
            fused = self.fusion_projs[i](fused)

            # ==========================================
            # 核心：通过 SFHM 模块进行 2D-FFT 频域强化和空域提纯
            # 这里会自动放大高频缺陷（划痕/裂纹），抑制低频背景
            # ==========================================
            sfhm_out = self.sfhm_modules[i](fused)

            enhanced_feas.append(sfhm_out)
            # 为了兼容你原有的 PFF (它期望收到两组特征去运算)
            # 我们直接把 SFHM 提纯后的特征映射回 256 维当作纯净的 ds_fea_adapted
            # 或者最简单的做法是：不用 PFF 里的 ds_fea_adapted 逻辑了，但为了少改代码，保持双路输入
        # ===============================================================

        # 4. 把经过 SFHM 极致强化的特征，送入你原有的 PFF 做最终金字塔融合
        # 注意：因为你的 PFF 是把 fea 和 ds_fea 再次拼接处理，
        # 为了不破坏你的原网络拓扑，fea 传入增强后的，ds_fea 依然传入原本的。
        # 这样 enhanced_feas 里的高频极度锐利，PFF 融合时会更依赖这些高频信息。
        final_fea = self.pff(enhanced_feas, ds_fea_adapted)

        # fea = self.pff(fea, ds_fea_adapted)

        '''# 计算 SFD (Semantic Frequency Differential)
        diff_maps = []
        for i in range(4):
            # 输入: FPN特征 (包含丰富细节), Adapter特征 (包含语义)
            # 输出: 差分图 (高频缺陷)
            d_map = self.sfd_modules[i](fea[i], ds_fea_adapted[i])
            diff_maps.append(d_map)'''

        # ==================== 执行可视化 ====================
        if VISUALIZE:
            # 为了防止生成太多图，我们只取尺度最大、细节最丰富的那一层（索引 0，对应 64x64 或 1/4 下采样层）
            save_feature_map(fea[0], "1_CNN_FPN_Layer0")

            # 因为原始 DINO 特征 [B, N, C] 是序列，你需要先把它 reshape 成二维才能画图
            # 你的代码里 patch_size 通常是 14 或 16，这里假设是 16。为了简单，我们直接看 adapted 之后的
            save_feature_map(ds_fea_adapted[0], "2_DINO_Adapted_Layer0")

            save_feature_map(enhanced_feas[0], "3_SFHM_Output_Layer0")
            save_feature_map(final_fea[0], "4_PFF_Final_Layer0")

            print("✅ 特征图已保存至 ./vis_results 目录！")
        # ====================================================

        # return fea, ds_fea, ds_fea_adapted
        return final_fea
        # return diff_maps

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

        '''diff_p2 = torch.abs(t1_p2 - t2_p2)
           diff_p3 = torch.abs(t1_p3 - t2_p3)
           diff_p4 = torch.abs(t1_p4 - t2_p4)
           diff_p5 = torch.abs(t1_p5 - t2_p5)'''

        # 2. 自顶向下处理（与原来类似，但没有diff计算）
        # 从最深层的p5开始
        fea_p5 = self.tb5(fea_p5)
        pred_p5 = self.p5_head(fea_p5)

        # p5特征融合到p4
        fea_p4 = self.p5_to_p4(fea_p5, fea_p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)

        # p4特征融合到p3
        fea_p3 = self.p4_to_p3(fea_p4, fea_p3)
        fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)

        # p3特征融合到p2
        fea_p2 = self.p3_to_p2(fea_p3, fea_p2)
        fea_p2 = self.tb2(fea_p2)
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
