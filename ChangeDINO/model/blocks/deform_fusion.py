# 文件路径: model/blocks/deform_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


'''class DeformableCrossAttentionFusion(nn.Module):
    """
    纯 PyTorch 空间缩减交叉注意力 (完美平替 C++ Deformable 算子)
    具备 O(N) 极低显存优势，且完美兼容 128 维(CNN)与 256 维(DINO)的高保真非对称融合。
    """

    def __init__(self, query_dim=128, value_dim=256, n_heads=4, kv_size=32):
        super().__init__()
        self.kv_size = kv_size  # 强制压缩 Key/Value 的空间分辨率上限，拯救显存

        # 核心：在 256 维的高保真空间内进行注意力交互
        self.attn = nn.MultiheadAttention(
            embed_dim=value_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # 升维投影层：将 CNN 从 128 维提拉到 256 维
        self.q_proj = nn.Conv2d(query_dim, value_dim, kernel_size=1)

        # 降维投影层：融合完毕后，压回 128 维送给后续解码器
        self.out_proj = nn.Conv2d(value_dim, query_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(query_dim)

        nn.init.constant_(self.out_proj.weight, 0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, query_feat, value_feat):
        B, C_q, H, W = query_feat.shape
        # value_feat 此时是包含极高语义的 256 维特征

        # 1. Query (CNN) 升维，且保持 100% 原始高分辨率，守住物理边缘！
        q_high = self.q_proj(query_feat)
        q = q_high.flatten(2).transpose(1, 2)  # [B, H*W, 256]

        # 2. Key/Value (DINO) 空间池化 (拯救显存的绝对关键)
        # 将 DINO 的空间分辨率强行池化到 32x32 (1024 个点)，消灭庞大的注意力矩阵
        if value_feat.shape[-1] > self.kv_size or value_feat.shape[-2] > self.kv_size:
            kv_feat = F.adaptive_avg_pool2d(value_feat, (self.kv_size, self.kv_size))
        else:
            kv_feat = value_feat

        k = kv_feat.flatten(2).transpose(1, 2)  # [B, 序列长度(如1024), 256]
        v = k

        # 3. 计算注意力 (由于 K/V 被大幅缩短，矩阵极小，瞬间计算完毕且不爆显存)
        attn_out, _ = self.attn(q, k, v)

        # 4. 还原回 2D 图像格式
        attn_out = attn_out.transpose(1, 2).reshape(B, value_feat.shape[1], H, W)

        # 5. 残差连接：原始 128 维 CNN 边缘 + 降维后的 128 维精准语义
        out = self.norm(query_feat + self.out_proj(attn_out))

        return out'''



class PurePyTorchDeformableAttn(nn.Module):
    """
    100% 纯 PyTorch 实现的可变形注意力机制 (基于 grid_sample)
    无需任何 C++ 编译，彻底告别报错！
    """

    def __init__(self, d_model=256, n_heads=4, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # 预测采样点的偏移量 (生成触手的方向和距离)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        # 预测各个采样点的重要性 (注意力权重)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        # 初始时，让触手不偏移(0)，权重均等分配
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, value, spatial_shapes):
        # query/value: [B, H*W, C], reference_points: [B, H*W, 1, 2]
        B, Lq, _ = query.shape
        H, W = spatial_shapes[0]

        value = self.value_proj(value)
        # 转换为 grid_sample 需要的空间形状
        value = value.view(B, H, W, self.n_heads, self.head_dim).permute(0, 3, 4, 1, 2)

        # 预测偏移量和注意力权重
        sampling_offsets = self.sampling_offsets(query).view(B, Lq, self.n_heads, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(B, Lq, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 将 [0,1] 范围的相对偏移量映射为实际坐标
        offset_normalizer = torch.tensor([W, H], dtype=query.dtype, device=query.device)
        sampling_locations = reference_points[:, :, :, None, :] \
                             + sampling_offsets / offset_normalizer[None, None, None, None, :]

        # 映射到 grid_sample 需要的 [-1, 1] 坐标系
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4)
        sampling_grids = 2 * sampling_locations - 1

        # ==== 核心：使用原生 grid_sample 进行稀疏采样，不爆显存 ====
        value_collapsed = value.reshape(B * self.n_heads, self.head_dim, H, W)
        sampling_grids = sampling_grids.reshape(B * self.n_heads, Lq, self.n_points, 2)

        sampled_value = F.grid_sample(value_collapsed, sampling_grids,
                                      mode='bilinear', padding_mode='zeros', align_corners=False)
        # ============================================================

        # 融合注意力权重
        sampled_value = sampled_value.view(B, self.n_heads, self.head_dim, Lq, self.n_points)
        attention_weights = attention_weights.permute(0, 2, 1, 3).unsqueeze(2)
        output = (sampled_value * attention_weights).sum(-1)
        output = output.permute(0, 3, 1, 2).reshape(B, Lq, self.d_model)

        return self.output_proj(output)


class DeformableCrossAttentionFusion(nn.Module):
    """
    高保真可变形交叉注意力融合模块 (包裹纯 PyTorch 算子)
    """

    def __init__(self, query_dim=128, value_dim=256, n_heads=4, n_points=4):
        super().__init__()

        # 实例化纯 PyTorch 的可变形注意力核心，运行在 256 维高保真空间
        self.deform_attn = PurePyTorchDeformableAttn(d_model=value_dim, n_heads=n_heads, n_points=n_points)

        # 升维：CNN 128 -> 256
        self.q_proj = nn.Conv2d(query_dim, value_dim, kernel_size=1)
        # 降维：融合后 256 -> 128
        self.out_proj = nn.Conv2d(value_dim, query_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(query_dim)

        # ====== 极其关键的零初始化 ======
        # 防止初始时随机噪声破坏物理边缘，使得 Epoch 1 也能平稳过渡
        nn.init.constant_(self.out_proj.weight, 0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def get_reference_points(self, H, W, device):
        # 生成归一化坐标点 [0, 1]
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref = torch.stack((ref_x, ref_y), -1)
        ref = ref[:, :, None, :]
        return ref

    def forward(self, query_feat, value_feat):
        B, C_q, H, W = query_feat.shape

        # 1. 升维 CNN，且保持 100% 原始分辨率
        q_high = self.q_proj(query_feat)

        # 2. 展平为序列 [B, H*W, 256]
        query = q_high.flatten(2).transpose(1, 2)
        value = value_feat.flatten(2).transpose(1, 2)

        spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=query_feat.device)
        reference_points = self.get_reference_points(H, W, query_feat.device).repeat(B, 1, 1, 1)

        # 3. 呼叫纯 PyTorch 版本的可变形采样！
        fused = self.deform_attn(
            query=query,
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes
        )

        # 4. 还原为图像，降回 128 维，加入残差
        fused_2d = fused.transpose(1, 2).reshape(B, value_feat.shape[1], H, W)
        out = self.norm(query_feat + self.out_proj(fused_2d))
        return out