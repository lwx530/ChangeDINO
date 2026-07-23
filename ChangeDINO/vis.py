import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入你的模型
from model.ChangeDINO import ChangeModel

import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np


def save_feature_map(feature_tensor, save_name, save_dir="vis_results_10_9", target_size=(256, 256)):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 递归处理多尺度特征（list/tuple类型）
    if isinstance(feature_tensor, (list, tuple)):
        for idx, feat in enumerate(feature_tensor):
            scale_name = f"{save_name}_scale{idx}"
            save_feature_map(feat, scale_name, save_dir, target_size)
        return

    # 2. 获取张量并脱离计算图放入 CPU
    feat = feature_tensor.detach().cpu()

    # 确保形状为 [B, C, H, W]
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)  # 如果没有 Batch 维度，补齐为 [1, C, H, W]
    elif feat.dim() != 4:
        print(f"  [跳过] {save_name} 不支持的张量维度: {feat.shape}")
        return

    # 3. 通道降维：将多通道特征沿着通道维度求均值，压缩为单通道特征图
    # 形状从 [B, C, H, W] 变为 [B, 1, H, W]
    feat_mean = torch.mean(feat, dim=1, keepdim=True)

    # 4. 尺度上采样：使用双线性插值统一调整为原始输入图像的分辨率 (256x256)
    feat_up = F.interpolate(feat_mean, size=target_size, mode='bilinear', align_corners=False)

    # 转为 2D NumPy 数组: [H, W]
    feat_np = feat_up.squeeze().numpy()

    # 5. 归一化：将张量内部的数值域线性映射到 0~1 区间，对齐颜色的对应范围
    f_min, f_max = feat_np.min(), feat_np.max()
    if f_max - f_min > 1e-8:
        feat_norm = (feat_np - f_min) / (f_max - f_min)
    else:
        feat_norm = np.zeros_like(feat_np)

    # 6. 伪彩色映射与保存
    plt.figure(figsize=(6, 6))

    # 使用 'jet' 颜色映射（蓝-绿-黄-红），并锁定值域在 0 到 1
    im = plt.imshow(feat_norm, cmap='jet', vmin=0, vmax=1)

    # 在图像右侧添加与论文图六中相同的 Colorbar 指示条
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # 关闭坐标轴
    plt.axis('off')

    # 保存结果（去除多余的白边）
    save_path = os.path.join(save_dir, f"{save_name}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载模型架构...")
    model = ChangeModel(backbone="resnet34").to(device)

    print("2. 正在加载训练好的权重...")
    # weight_path = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-36/ESDI-36_resnet34_best.pth"
    weight_path = "/root/autodl-tmp/ChangeDINO/checkpoints/ESDI-40/ESDI-40_resnet34_best.pth"

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


    print("4. 正在读取并预处理图片...")
    # img_path = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test/images/10_28.jpg"
    img_path = "/root/autodl-tmp/ChangeDINO/datasets/ESDIs-SOD/test/images/10_9.jpg"

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
    save_dir = "vis_results_10_9"
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