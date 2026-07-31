
from PIL import Image
import torchvision.transforms as transforms

# 导入你的模型
from model.ChangeDINO import ChangeModel

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np


def save_feature_map(feature_tensor, save_name, save_dir="vis_results_10_11", target_size=(256, 256)):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 递归处理多尺度特征
    if isinstance(feature_tensor, (list, tuple)):
        for idx, feat in enumerate(feature_tensor):
            scale_name = f"{save_name}_scale{idx}"
            save_feature_map(feat, scale_name, save_dir, target_size)
        return

    # 2. 获取张量并放入 CPU
    feat = feature_tensor.detach().cpu()
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    elif feat.dim() != 4:
        print(f"  [跳过] {save_name} 不支持的张量维度: {feat.shape}")
        return

    # 3. 先进行上采样，对齐到 256x256
    feat_up = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
    feat_up = feat_up.squeeze(0)  # 变为 [C, H, W]

    C, H, W = feat_up.shape

    # 4. 智能筛选策略：计算每个通道的方差，找出波动最大（信息最丰富）的 Top-16
    feat_flat = feat_up.view(C, -1)
    variances = torch.var(feat_flat, dim=1)  # 计算方差

    # 防止通道数不足 16 的情况（比如最后的 p1 输出通常只有单通道或少通道）
    n = min(C, 16)
    _, topk_indices = torch.topk(variances, n)

    # 取出这 16 个“偏科”通道
    selected_features = feat_up[topk_indices]

    # 5. 绘制网格图 (改为 4x4 矩阵排列)
    rows = int(np.ceil(n / 4))  # 计算行数，每行最多 4 张
    cols = min(n, 4)  # 计算列数
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # 兼容处理：把二维的 axes 数组展平为一维，方便后续循环
    if n > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(n):
        fm = selected_features[i].numpy()

        # 对单个通道进行 Min-Max 归一化
        f_min, f_max = fm.min(), fm.max()
        if f_max - f_min > 1e-8:
            fm_norm = (fm - f_min) / (f_max - f_min)
        else:
            fm_norm = np.zeros_like(fm)

        # 使用 'jet' 恢复经典的论文蓝红热力图风格
        axes[i].imshow(fm_norm, cmap='jet')
        axes[i].axis('off')
        axes[i].set_title(f"Ch: {topk_indices[i].item()}", fontsize=10)

    # 把没画满的多余子图隐藏掉
    for j in range(n, len(axes)):
        axes[j].axis('off')
    # 6. 紧凑排版并保存
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{save_name}_top{n}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载模型架构...")
    model = ChangeModel(backbone="resnet34").to(device)

    print("2. 正在加载训练好的权重...")
    weight_path = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-4/ESDI-4_resnet34_best.pth"
    # weight_path = "/root/autodl-tmp/ChangeDINO/checkpoints/ESDI-1/ESDI-1_resnet34_best.pth"

    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get('network',checkpoint.get('model_state_dict',checkpoint))
        model.load_state_dict(state_dict, strict=False)
        print("加载权重成功！")
    else:
        print(f"警告：未找到权重文件{weight_path}")

    model.eval()

    print("3. 正在注册模块特征提取 Hook...")
    all_features = {}
    hook_handles = []

    def get_hook(name):
        def hook(module, input, output):
            all_features[name] = output
        return hook

    # 将钩子挂载到模型内部的 dino 模块上
    hook_handles.append(model.encoder.backbone.register_forward_hook(get_hook('1_resnet34')))

    hook_handles.append(model.encoder.dino.register_forward_hook(get_hook('2_dino')))
    hook_handles.append(model.encoder.groupweight.register_forward_hook(get_hook('3_groupweight')))
    hook_handles.append(model.encoder.defect_adapter.register_forward_hook(get_hook('4_defect_adapter')))

    hook_handles.append(model.encoder.sfhm_modules[0].register_forward_hook(get_hook('5_SFHM0')))

    hook_handles.append(model.encoder.pff.register_forward_hook(get_hook('6_pff')))

    hook_handles.append(model.detector.tb1.register_forward_hook(get_hook('7_tb1')))

    hook_handles.append(model.detector.p1_head.register_forward_hook(get_hook('8_p1')))

    hook_handles.append(model.detector.edge.register_forward_hook(get_hook('9_edge')))


    print("4. 正在读取并预处理图片...")
    img_path = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test/images/10_11.jpg"
    # img_path = "/root/autodl-tmp/ChangeDINO/datasets/ESDIs-SOD/test/images/10_11.jpg"

    if not os.path.exists(img_path):
        print(f"❌ 找不到图片：{img_path}")
        return

    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    print("5. 正在进行前向推理...")
    with torch.no_grad():
        model._forward(input_tensor)

    print("6. 正在绘制特征热力图...")
    # 推理完成后，dino_features 字典里已经装满了我们要的特征
    save_dir = "vis_results_10_11"
    for layer_name in sorted(all_features.keys()):
        print(f"   -> 保存 {layer_name} 特征图...")
        feat_data = all_features[layer_name]
        save_feature_map(feat_data, layer_name, save_dir=save_dir)

    # 用完后拆除钩子
    for handle in hook_handles:
        handle.remove()

    print(f"\n🎉 大功告成！请去工程目录下的 {save_dir} 文件夹查看特征热力图！")

if __name__ == "__main__":
    main()