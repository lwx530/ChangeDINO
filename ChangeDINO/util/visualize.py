import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 关闭matplotlib交互模式，适配测试阶段批量运行
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def denorm_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    denorm_img = img_tensor * std + mean
    return torch.clamp(denorm_img, 0.0, 1.0)


def process_feature_map(feat, mode="max_pool"):
    """
    替换全局平均为「最大池化/缺陷通道筛选」，减少缺陷响应稀释
    :param feat: 特征张量 [C, H, W]，C=1024（DINOv3）/256（Adapter）
    :param mode: 特征聚合方式：max_pool（最大池化，保留最强响应）/mean（原平均，对比用）
    :return: 可视化特征图 [H, W]，np.uint8，值范围[0,255]
    """
    feat = feat.detach().cpu()
    # 核心修改：用最大池化替代全局平均，保留每个空间位置的最强通道响应
    if mode == "max_pool":
        feat = torch.max(feat, dim=0, keepdim=False)[0]  # [C,H,W]→[H,W]，取每个位置的最大通道响应
    else:
        feat = torch.mean(feat, dim=0, keepdim=False)  # 原全局平均

    # 归一化保持不变
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    feat = (feat * 255).numpy().astype(np.uint8)
    return feat

def resize_feature_to_ori(feat, ori_size):
    feat_tensor = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
    feat_resize = F.interpolate(
        feat_tensor, size=ori_size, mode="bilinear", align_corners=False
    )
    return feat_resize.squeeze().numpy().astype(np.uint8)

def visualize_dino_adapter(
    img_ori,
    dino_feats,
    adapter_feats,
    save_dir,
    img_name,
    dino_sizes=[64,32,16,8],
    cmap='viridis'
):
    # 预处理原图
    img_denorm = denorm_image(img_ori)
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    ori_h, ori_w = img_np.shape[:2]
    os.makedirs(save_dir, exist_ok=True)

    # 预处理DINO和Adapter特征
    dino_vis_feats = [resize_feature_to_ori(process_feature_map(f, mode="max_pool"), (ori_h, ori_w)) for f in dino_feats]
    adapter_vis_feats = [resize_feature_to_ori(process_feature_map(f, mode="max_pool"), (ori_h, ori_w)) for f in adapter_feats]

    # ========== 核心修复：固定画布尺寸+自适应子图，避免宽高比失衡 ==========
    # 固定画布宽度为12（适配4列特征），高度按原图比例计算
    fig_width = 12
    fig_height = fig_width * (ori_h / (ori_w * 4)) * 3  # 3行 × 单图比例
    fig, axes = plt.subplots(3, 5, figsize=(fig_width, fig_height), dpi=150)
    # fig.suptitle(f'Feature Visualization - {img_name}', fontsize=12, y=0.98)
    # 缩小间隔到极致（几乎无空白）
    plt.subplots_adjust(hspace=0.0, wspace=0.0, top=0.95, bottom=0.05, left=0.1, right=0.95)

    # 第一列：左侧标注
    axes[0, 0].text(0.5, 0.5, 'Original', ha='center', va='center', fontsize=10, rotation=0)
    axes[1, 0].text(0.5, 0.5, 'DINOv3', ha='center', va='center', fontsize=10, rotation=0)
    axes[2, 0].text(0.5, 0.5, 'Adapter', ha='center', va='center', fontsize=10, rotation=0)
    for i in range(3):
        axes[i, 0].axis('off')
        axes[i, 0].set_facecolor('white')

    # 右侧4列：特征图（强制填充子图区域）
    for col in range(4):
        # 原图行
        axes[0, col+1].imshow(img_np)
        axes[0, col + 1].set_title(f'Scale {col + 1}\n({dino_sizes[col]}×{dino_sizes[col]})', fontsize=10)
        axes[0, col+1].axis('off')
        axes[0, col+1].set_aspect('auto')  # 自适应填充子图

        # DINOv3行
        axes[1, col+1].imshow(dino_vis_feats[col], cmap=cmap)
        axes[1, col+1].axis('off')
        axes[1, col+1].set_aspect('auto')

        # Adapter行
        axes[2, col+1].imshow(adapter_vis_feats[col], cmap=cmap)
        axes[2, col+1].axis('off')
        axes[2, col+1].set_aspect('auto')

    # 保存（强制裁剪空白）
    save_path = os.path.join(save_dir, f'{img_name}vis.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=150)
    plt.close(fig)
    print(f"✅ 可视化结果已保存：{save_path}")


if __name__ == "__main__":
    img_ori = torch.randn(3, 256, 256)
    dino_feats = [torch.randn(1024, 64, 64), torch.randn(1024, 32, 32),
                  torch.randn(1024, 16, 16), torch.randn(1024, 8, 8)]
    adapter_feats = [torch.randn(256, 64, 64), torch.randn(256, 32, 32),
                     torch.randn(256, 16, 16), torch.randn(256, 8, 8)]
    visualize_dino_adapter(
        img_ori=img_ori,
        dino_feats=dino_feats,
        adapter_feats=adapter_feats,
        save_dir='./test_vis',
        img_name='test_sample'
    )