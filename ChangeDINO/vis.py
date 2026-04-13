import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入你的模型
from model.ChangeDINO import ChangeModel


def save_feature_map(feature_tensor, save_name, save_dir="vis_results_dino"):
    """
    使用 PCA (主成分分析) 提取 DINOv3 语义特征并保存为热力图
    """
    os.makedirs(save_dir, exist_ok=True)
    if feature_tensor.dim() == 4:
        feat = feature_tensor[0]  # 取 Batch 中的第一张图 [C, H, W]
    else:
        feat = feature_tensor

    C, H, W = feat.shape

    # 1. 将特征展平: [C, H, W] -> [H*W, C]
    feat_flat = feat.view(C, -1).permute(1, 0)

    # 为了让 PCA 更准确，先减去空间均值
    feat_flat = feat_flat - feat_flat.mean(dim=0, keepdim=True)

    # 2. 使用 PyTorch 自带的 PCA 提取第一主成分
    # q=1 表示只提取最重要的 1 个维度，它代表了最强烈的语义聚集区域
    U, S, V = torch.pca_lowrank(feat_flat, q=1)

    # 投影到第一主成分，并变回图像形状 [H, W]
    feat_pca = torch.matmul(feat_flat, V[:, :1]).view(H, W)

    feat_map = feat_pca.detach().cpu().numpy()

    # 3. 归一化到 0-1 区间
    feat_map = (feat_map - np.min(feat_map)) / (np.max(feat_map) - np.min(feat_map) + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(feat_map, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title(save_name)
    plt.savefig(os.path.join(save_dir, f"{save_name}.png"), bbox_inches='tight')
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载模型架构...")
    model = ChangeModel(backbone="mobilenetv2").to(device)

    print("2. 正在加载训练好的权重...")
    weight_path = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-4/ESDI-4_mobilenetv2_best.pth"

    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        # 【Bug 修复】：你的 create_ChangeDINO.py 中保存的 key 是 'network'，不是 'model_state_dict'！
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

    print("3. 正在注册 DINOv3 特征提取 Hook...")
    # 建立一个字典，用来自动“接住” DINO 流出的特征
    dino_features = {}

    def hook_fn(module, input, output):
        # 根据你的 ChangeDINO.py，dino(x) 的 output 是一个包含 24 层张量的 list
        # 我们截获 5, 11, 17, 23 层的特征 (注意：0-indexed，所以索引就是对应的数字)
        dino_features['Layer_05'] = output[5]
        dino_features['Layer_11'] = output[11]
        dino_features['Layer_17'] = output[17]
        dino_features['Layer_23'] = output[23]

    # 将钩子挂载到模型内部的 dino 模块上
    hook_handle = model.encoder.dino.register_forward_hook(hook_fn)

    print("4. 正在读取并预处理图片...")
    img_path = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test/images/10_1.jpg"

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
    save_dir = "vis_results_dino"
    for layer_name, feat_tensor in dino_features.items():
        print(f"   -> 保存 {layer_name} 特征图...")
        save_feature_map(feat_tensor, f"InnerAdapter_{layer_name}", save_dir=save_dir)

    # 用完后拆除钩子
    hook_handle.remove()

    print(f"\n🎉 大功告成！请去工程目录下的 {save_dir} 文件夹查看特征热力图！")


if __name__ == "__main__":
    main()