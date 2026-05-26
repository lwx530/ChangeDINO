import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入你的模型
from model.ChangeDINO import ChangeModel


def save_feature_map(feature_tensor, save_name, save_dir="vis_results_1_57"):
    """
    使用 RGB-PCA 提取 DINOv3 丰富语义特征并保存为彩色图
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(feature_tensor, (list,tuple)):
        for idx,feat in enumerate(feature_tensor):
            scale_name = f"{save_name}_scale{idx}"
            save_feature_map(feat, scale_name, save_dir)
        return

    if feature_tensor.dim() == 4:
        feat = feature_tensor[0]  # [C, H, W]
    else:
        feat = feature_tensor

    C, H, W = feat.shape

    if C == 1 or C == 2:
        for i in range(C):
            feat_map = feat[0].detach().cpu().numpy()
            plt.figure(figsize=(6,6))
            plt.imshow(feat_map, cmap='jet')
            plt.axis('off')
            title = f"{save_name}_Channel_{i}" if C == 2 else f"{save_name}_Mask"
            plt.title(title)
            save_path = os.path.join(save_dir, f"{title}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        return

    # 1. 展平并去中心化
    feat_flat = feat.view(C, -1).permute(1, 0)
    feat_flat = feat_flat - feat_flat.mean(dim=0, keepdim=True)

    # 2. 提取前 3 个主成分 (PC1, PC2, PC3)
    feat_flat = feat_flat + torch.randn_like(feat_flat) * 1e-5
    try:
        U, S, V = torch.pca_lowrank(feat_flat, q=3)
        feat_pca = torch.matmul(feat_flat, V[:, :3])  # [H*W, 3]
        feat_pca = feat_pca.view(H, W, 3).detach().cpu().numpy()

    # 3. 分别将三个通道归一化到 0-1 区间，映射为 RGB
        for i in range(3):
            comp = feat_pca[:, :, i]
            comp_min, comp_max = comp.min(), comp.max()
            feat_pca[:, :, i] = (comp - comp_min) / (comp_max - comp_min + 1e-8)

    # 画图并保存热力图
        plt.figure(figsize=(6, 6))
        plt.imshow(feat_pca)  # 直接显示 RGB 彩色图
        plt.axis('off')
        plt.title(f"{save_name}_RGB_PCA")
        plt.savefig(os.path.join(save_dir, f"{save_name}_rgb.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [跳过]{save_name} PCA失败： {e}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载模型架构...")
    model = ChangeModel(backbone="mobilenetv2").to(device)

    print("2. 正在加载训练好的权重...")
    weight_path = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-20/ESDI-20_mobilenetv2_best.pth"

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
    hook_handles.append(model.encoder.fpn.register_forward_hook(get_hook('1_CNN_FPN')))

    hook_handles.append(model.encoder.dino.register_forward_hook(get_hook('2_dino')))

    hook_handles.append(model.encoder.defect_adapter.register_forward_hook(get_hook('3_defect_adapter')))

    # hook_handles.append(model.encoder.sfhm_modules[0].register_forward_hook(get_hook('4_sfhm_module')))

    hook_handles.append(model.encoder.pff.register_forward_hook(get_hook('5_PFF')))

    # hook_handles.append(model.encoder.srf_mask_gen.register_forward_hook(get_hook('6_srf_mask_gen')))

    hook_handles.append(model.detector.tb5.register_forward_hook(get_hook('7_detector_tb5')))
    hook_handles.append(model.detector.tb4.register_forward_hook(get_hook('7_detector_tb4')))
    hook_handles.append(model.detector.tb3.register_forward_hook(get_hook('7_detector_tb3')))
    hook_handles.append(model.detector.tb2.register_forward_hook(get_hook('7_detector_tb2')))

    hook_handles.append(model.refiner.register_forward_hook(get_hook('8_LMM')))

    print("4. 正在读取并预处理图片...")
    img_path = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test/images/1_57.jpg"

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
    save_dir = "vis_results_1_57"
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