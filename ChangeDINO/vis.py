import os
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from model.ChangeDINO import ChangeModel

# ==================== 配置 ====================
WEIGHT_PATH = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-24/ESDI-24_resnet34_best.pth"
DATA_ROOT = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test"
IMG_SIZE = 256
SAVE_ROOT = "vis_results"

# 测试样本名称
IMG_NAMES = ["7_34", "1_34", "6_30", "10_52"]


# =============================================


def tensor_to_heatmap_np(feat_tensor, target_size=(256, 256)):
    """
    将任意维度的特征图统一转为 [0, 1] 的 HxW 热力图 numpy 数组
    """
    feat = feat_tensor.detach().cpu().float()
    if feat.dim() == 4:
        feat = feat[0]  # [C, H, W]

    if feat.dim() == 3:
        if feat.shape[0] == 1:
            heat = feat[0]
        else:
            # 通道 L2 范数（能量响应）
            heat = torch.norm(feat, p=2, dim=0)
    elif feat.dim() == 2:
        heat = feat
    else:
        return None

    heat = heat.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    heat = F.interpolate(heat, size=target_size, mode="bilinear", align_corners=False)[0, 0]
    heat_np = heat.numpy()

    # 归一化至 0~1
    heat_np = (heat_np - heat_np.min()) / (heat_np.max() - heat_np.min() + 1e-8)
    return heat_np


def save_feature_heatmap(feature_obj, save_name, save_dir="vis", target_size=(256, 256)):
    """
    递归保存单个或多尺度特征图
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(feature_obj, (list, tuple)):
        for idx, feat in enumerate(feature_obj):
            if isinstance(feat, torch.Tensor):
                save_feature_heatmap(feat, f"{save_name}_scale{idx + 1}", save_dir, target_size)
        return

    if isinstance(feature_obj, torch.Tensor):
        heat = tensor_to_heatmap_np(feature_obj, target_size)
        if heat is not None:
            plt.imsave(os.path.join(save_dir, f"{save_name}.png"), heat, cmap="jet")


def thin_boundary(mask_t):
    """mask_t: [1,1,H,W] {0,1} -> 真实 1 像素物理边缘提取"""
    return F.max_pool2d(mask_t, 3, 1, 1) - (-F.max_pool2d(-mask_t, 3, 1, 1))


def diagnose_one(model, name, feats, device):
    img_root, gt_root = os.path.join(DATA_ROOT, "images"), os.path.join(DATA_ROOT, "gt")

    cand = glob.glob(os.path.join(img_root, name + ".*"))
    if not cand:
        print(f"[跳过] 找不到图片: {name}")
        return
    img_path = cand[0]
    gt_path = os.path.join(gt_root, name + ".png")

    if not os.path.exists(gt_path):
        print(f"[跳过] 找不到标签: {gt_path}")
        return

    save_dir = os.path.join(SAVE_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)

    # ---- 图像输入预处理 ----
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    x = tf(img).unsqueeze(0).to(device)

    # ---- GT 加载与边界提取 ----
    gt_pil = Image.open(gt_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    gt_np = (np.array(gt_pil) > 127).astype(np.float32)
    gt_t = torch.from_numpy(gt_np)[None, None].to(device)
    gb = thin_boundary(gt_t)

    # ---- 模型前向传播 ----
    feats.clear()
    model.eval()
    with torch.no_grad():
        # ChangeModel forward 返回 Se, Sm
        pred_se, pred_sm = model(x)

    # ==================== 统计信息输出 ====================
    print("\n" + "=" * 64)
    print(f"样本分析: {name}")
    print("=" * 64)

    prob_se = torch.sigmoid(pred_se)
    prob_sm = torch.sigmoid(pred_sm)

    print("[GT]       前景比例            = %.4f" % gt_t.mean().item())
    print("[Sm 粗预测] 激活区域占比 (>0.5) = %.4f | Sigmoid 均值 = %.4f" %
          ((prob_sm > 0.5).float().mean().item(), prob_sm.mean().item()))
    print("[Se 精炼预测] 激活区域占比 (>0.5)= %.4f | Sigmoid 均值 = %.4f" %
          ((prob_se > 0.5).float().mean().item(), prob_se.mean().item()))

    tgt = (IMG_SIZE, IMG_SIZE)

    # ==================== 核心诊断图排版 (8 联画) ====================
    # 观察整个特征传递与精炼过程
    fig, ax = plt.subplots(1, 8, figsize=(32, 4.2))

    # 1. 原图
    ax[0].imshow(img.resize(tgt))
    ax[0].set_title("1. Input Image")

    # 2. GT 真实标签
    ax[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("2. Ground Truth")

    # 3. DINO 顶层适配特征
    if "02_adapter" in feats and isinstance(feats["02_adapter"], (list, tuple)):
        # 提取 DINO 适配后的最小尺度（最深层语义，通常在 -1 或 0）
        dino_heat = tensor_to_heatmap_np(feats["02_adapter"][-1], tgt)
        ax[2].imshow(dino_heat, cmap="jet")
    ax[2].set_title("3. DINO Top Semantics")

    # 4. LGA 跨模态融合后 (尺度1，最大分辨率/保留细节)
    if "03_lga_fused" in feats and isinstance(feats["03_lga_fused"], (list, tuple)):
        lga_heat = tensor_to_heatmap_np(feats["03_lga_fused"][0], tgt)
        ax[3].imshow(lga_heat, cmap="jet")
    ax[3].set_title("4. LGA Fused (Scale 1)")

    # 5. 解码器最终输出特征 Fs'
    if "07_dec_fea1D" in feats:
        fs_heat = tensor_to_heatmap_np(feats["07_dec_fea1D"], tgt)
        ax[4].imshow(fs_heat, cmap="jet")
    ax[4].set_title("5. Decoder Feat (Fs')")

    # 6. EGRefiner 高频边缘特征 (fe_cat / post_edge)
    if "09_egrefiner_edge" in feats:
        edge_heat = tensor_to_heatmap_np(feats["09_egrefiner_edge"], tgt)
        ax[5].imshow(edge_heat, cmap="jet")
    ax[5].set_title("6. EGRefiner Edge")

    # 7. Sm 粗预测结果
    sm_np = F.interpolate(prob_sm, size=tgt, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    ax[6].imshow(sm_np, cmap="jet", vmin=0, vmax=1)
    ax[6].set_title("7. Pred Sm (Aux Head)")

    # 8. Se 最终精炼预测
    se_np = F.interpolate(prob_se, size=tgt, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    ax[7].imshow(se_np, cmap="jet", vmin=0, vmax=1)
    ax[7].set_title("8. Pred Se (Final)")

    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pipeline_diagnosis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ==================== 保存各模块所有尺度的单独热力图 ====================
    for key, feat in feats.items():
        save_feature_heatmap(feat, save_name=key, save_dir=save_dir, target_size=tgt)

    print(f"-> 诊断图与中间特征热力图已保存至: {save_dir}/")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 构建 ChangeModel 模型...")
    # 与你的 ChangeModel 定义一致（fpn_channels=64）
    model = ChangeModel(backbone="resnet34", fpn_channels=64).to(device)

    print("2. 加载权重...")
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 找不到权重文件: {WEIGHT_PATH}")
        return

    ckpt = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get("network", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"   权重加载情况: 缺失 key: {len(missing)} 个, 冗余 key: {len(unexpected)} 个")
    if missing:
        print("   部分 missing keys:", missing[:3])

    print("3. 精准挂载关键模块 Hook...")
    feats = {}
    handles = []

    def get_hook(name):
        def hook(module, inp, out):
            feats[name] = out

        return hook

    # --- 编码器关键阶段 ---
    # 1. CNN 主干输出 (4 尺度)
    handles.append(model.encoder.backbone.register_forward_hook(get_hook("01_backbone_cnn")))
    # 2. DINO 经 Defect Adapter 降维与多尺度投影后的特征
    handles.append(model.encoder.defect_adapter.register_forward_hook(get_hook("02_adapter")))
    # 3. LGA 跨模态融合模块输出 (x1_out, x2_out, x3_out, x4_out)
    handles.append(model.encoder.pff.register_forward_hook(get_hook("03_lga_fused")))

    # --- 解码器关键阶段 ---
    # 4. 深层 TransformerBlock 注意力模块
    handles.append(model.decoder.tb4.register_forward_hook(get_hook("04_dec_tb4")))
    handles.append(model.decoder.tb3.register_forward_hook(get_hook("05_dec_tb3")))
    # 5. 浅层平滑卷积
    handles.append(model.decoder.convD2.register_forward_hook(get_hook("06_dec_convD2")))
    handles.append(model.decoder.convD1.register_forward_hook(get_hook("07_dec_fea1D")))

    # --- 高频精炼与预测头 ---
    # 6. EGRefiner 中的多尺度边缘卷积融合结果
    handles.append(model.decoder.refiner.post_edge.register_forward_hook(get_hook("09_egrefiner_edge")))
    # 7. 辅助预测头 Sm
    handles.append(model.decoder.aux_head.register_forward_hook(get_hook("10_aux_head_Sm")))

    print(f"4. 开始分析 {len(IMG_NAMES)} 张图像...")
    for name in IMG_NAMES:
        diagnose_one(model, name, feats, device)

    for h in handles:
        h.remove()

    print(f"\n🎉 诊断完成！结果已存放在: {SAVE_ROOT}/")


if __name__ == "__main__":
    main()