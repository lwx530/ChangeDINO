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
WEIGHT_PATH = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-7/ESDI-7_resnet34_best.pth"
DATA_ROOT   = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test"
IMG_SIZE    = 256
SAVE_ROOT   = "vis_results"

# 想多看几张就加名字（不带扩展名）
IMG_NAMES = ["1_8","1_19","7_4","10_9"]
# =============================================


def save_feature_heatmap(feature_tensor, image_pil, save_name, save_dir="vis", target_size=(256, 256)):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(feature_tensor, (list, tuple)):
        for idx, feat in enumerate(feature_tensor):
            save_feature_heatmap(feat, image_pil, f"{save_name}_scale{idx}", save_dir, target_size)
        return

    feat = feature_tensor.detach().cpu()
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    if feat.dim() != 4:
        return

    feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)

    if feat.shape[1] == 1:
        heat = feat[0, 0]
    else:
        heat = torch.norm(feat[0], p=2, dim=0)

    heat = heat.numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    plt.imsave(os.path.join(save_dir, f"{save_name}_heat.png"), heat, cmap="jet")


def thin_boundary(mask_t):
    """mask_t: [1,1,H,W] {0,1} -> 形态学梯度，严格 1 像素边界"""
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

    # ---- 输入 ----
    img = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    x = tf(img).unsqueeze(0).to(device)

    # ---- GT（缩到网络网格，和训练时的目标一致）----
    gt_pil = Image.open(gt_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    gt_np = (np.array(gt_pil) > 127).astype(np.float32)
    gt_t = torch.from_numpy(gt_np)[None, None].to(device)
    gb = thin_boundary(gt_t)

    # ---- 前向 ----
    feats.clear()
    model.eval()
    with torch.no_grad():
        pred_main = model._forward(x)

    # ==================== 统计 ====================
    print("\n" + "=" * 64)
    print(f"图片: {name}")
    print("=" * 64)

    if "11_boundary" in feats:
        bp = torch.sigmoid(feats["11_boundary"])
        print("[pred boundary] sigmoid mean = %.4f" % bp.mean().item())
        print("[pred boundary] frac > 0.5   = %.4f" % (bp > 0.5).float().mean().item())
        print("[pred boundary] min / max    = %.4f / %.4f" % (bp.min().item(), bp.max().item()))
    else:
        print("[警告] 没抓到 11_boundary，检查 hook 是否挂上 decoder.conv5")

    print("[GT]   foreground frac       = %.4f" % gt_t.mean().item())
    print("[GT]   thin-boundary frac    = %.4f" % gb.mean().item())

    pm = torch.sigmoid(pred_main)
    print("[main] pred frac > 0.5       = %.4f" % (pm > 0.5).float().mean().item())
    print("[main] pred mean             = %.4f" % pm.mean().item())

    if "9_edge" in feats and "9_edge_in" in feats:
        ein = feats["9_edge_in"][0]
        eout = feats["9_edge"][0]
        a = eout.norm(dim=0).flatten()
        b = ein.norm(dim=0).flatten()
        a = a - a.mean()
        b = b - b.mean()
        corr = (a @ b / (a.norm() * b.norm() + 1e-8)).item()
        print("[edge] corr(edge_out, edge_in) = %.4f" % corr)
    print("=" * 64)

    # ==================== 五联图 ====================
    tgt = (IMG_SIZE, IMG_SIZE)

    pred_mask = F.interpolate(pm, size=tgt, mode="bilinear", align_corners=False)[0, 0]
    pred_b = F.interpolate(torch.sigmoid(feats["11_boundary"]), size=tgt,
                           mode="bilinear", align_corners=False)[0, 0]

    e_out = feats["9_edge"][0].norm(dim=0, keepdim=True).unsqueeze(0)
    e_out = F.interpolate(e_out, size=tgt, mode="bilinear", align_corners=False)[0, 0]
    e_out = (e_out - e_out.min()) / (e_out.max() - e_out.min() + 1e-8)

    e_in = feats["9_edge_in"][0].norm(dim=0, keepdim=True).unsqueeze(0)
    e_in = F.interpolate(e_in, size=tgt, mode="bilinear", align_corners=False)[0, 0]
    e_in = (e_in - e_in.min()) / (e_in.max() - e_in.min() + 1e-8)

    fig, ax = plt.subplots(1, 6, figsize=(26, 4.6))
    ax[0].imshow(img.resize(tgt));                                     ax[0].set_title("image")
    ax[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1);                  ax[1].set_title("GT mask")
    ax[2].imshow(gb[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1); ax[2].set_title("GT boundary (1px)")
    ax[3].imshow(pred_b.cpu().numpy(), cmap="jet", vmin=0, vmax=1);    ax[3].set_title("pred boundary (sigmoid)")
    ax[4].imshow(pred_mask.cpu().numpy(), cmap="jet", vmin=0, vmax=1); ax[4].set_title("pred mask")
    ax[5].imshow(e_out.cpu().numpy(), cmap="jet");                     ax[5].set_title("edge_out energy")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "boundary_diagnosis.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # ==================== 常规热力图 ====================
    for fname in sorted(feats.keys()):
        if fname.endswith("_in") or fname == "11_boundary":
            continue
        save_feature_heatmap(feats[fname], img, fname, save_dir=save_dir, target_size=tgt)

    print(f"-> 已保存到 {save_dir}/")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 构建模型...")
    model = ChangeModel(backbone="resnet34").to(device)

    print("2. 加载权重...")
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 找不到权重: {WEIGHT_PATH}")
        return
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    state_dict = ckpt.get("network", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"   缺失的 key: {len(missing)}  多余的 key: {len(unexpected)}")
    if missing:
        print("   missing[:5] =", missing[:5])
    if unexpected:
        print("   unexpected[:5] =", unexpected[:5])

    print("3. 注册 hook...")
    feats = {}
    handles = []

    def get_hook(name):
        def hook(module, inp, out):
            feats[name] = out
            if isinstance(inp, (tuple, list)) and len(inp) > 0:
                feats[name + "_in"] = inp[0]
        return hook

    handles.append(model.encoder.backbone.register_forward_hook(get_hook("1_resnet34")))
    handles.append(model.encoder.dino.register_forward_hook(get_hook("2_dino")))
    handles.append(model.encoder.defect_adapter.register_forward_hook(get_hook("4_defect_adapter")))
    handles.append(model.decoder.edge.register_forward_hook(get_hook("9_edge")))
    handles.append(model.decoder.conv5.register_forward_hook(get_hook("11_boundary")))   # ★ 新增
    handles.append(model.decoder.conv4.register_forward_hook(get_hook("10_edge4")))
    handles.append(model.decoder.conv3.register_forward_hook(get_hook("10_edge3")))
    handles.append(model.decoder.conv2.register_forward_hook(get_hook("10_edge2")))
    handles.append(model.decoder.conv1.register_forward_hook(get_hook("10_edge1")))

    print(f"4. 开始诊断 {len(IMG_NAMES)} 张图...")
    for name in IMG_NAMES:
        diagnose_one(model, name, feats, device)

    for h in handles:
        h.remove()

    print(f"\n🎉 完成，去看 {SAVE_ROOT}/")


if __name__ == "__main__":
    main()