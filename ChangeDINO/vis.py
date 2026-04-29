import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入你的模型
from model.ChangeDINO import ChangeModel


def save_feature_map(feature_tensor, save_name, save_dir="vis_results_cda-1_15"):
    """
    使用 RGB-PCA 提取 DINOv3 丰富语义特征并保存为彩色图
    """
    os.makedirs(save_dir, exist_ok=True)

    if feature_tensor.dim() == 4:
        feat = feature_tensor[0]  # [C, H, W]
    else:
        feat = feature_tensor

    C, H, W = feat.shape

    # 1. 展平并去中心化
    feat_flat = feat.view(C, -1).permute(1, 0)
    feat_flat = feat_flat - feat_flat.mean(dim=0, keepdim=True)

    # 2. 提取前 3 个主成分 (PC1, PC2, PC3)
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
    plt.title(save_name + "_RGB")
    plt.savefig(os.path.join(save_dir, f"{save_name}_rgb.png"), bbox_inches='tight')
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载模型架构...")
    model = ChangeModel(backbone="mobilenetv2").to(device)

    print("2. 正在加载训练好的权重...")
    weight_path = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-13/ESDI-13_mobilenetv2_best.pth"

    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        if 'network' in checkpoint:
            state_dict = checkpoint['network']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功！")
    else:
        print(f"❌ 警告：未找到权重文件 {weight_path}")

    model.eval()

    print("3. 正在注册 CDA 模块特征提取 Hook...")
    cda_features = {}

    def hook_fn_cda(module, input, output):
        # input 是一个元组，input[0] 就是进入 tb2 (CDA) 之前的特征
        cda_features['1_Before_CDA'] = input[0]
        # output 是经过 tb2 (CDA) 处理后的特征
        cda_features['2_After_CDA'] = output

    # 将钩子挂载到模型内部的 dino 模块上
    hook_handle = model.detector.tb2.register_forward_hook(hook_fn_cda)

    print("4. 正在读取并预处理图片...")
    img_path = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test/images/1_15.jpg"

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
    save_dir = "vis_results_cda-1_15"
    for layer_name, feat_tensor in cda_features.items():
        print(f"   -> 保存 {layer_name} 特征图...")
        save_feature_map(feat_tensor, f"{layer_name}", save_dir=save_dir)

    # 用完后拆除钩子
    hook_handle.remove()

    print(f"\n🎉 大功告成！请去工程目录下的 {save_dir} 文件夹查看特征热力图！")

if __name__ == "__main__":
    main()