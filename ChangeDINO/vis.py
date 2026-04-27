import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入你的模型
from model.ChangeDINO import ChangeModel


def save_logit_map(logit_tensor, save_name, save_dir="vis_results_lmm-7_14"):
    """
    针对 LMM 模块的 2 通道 Logits 进行可视化。
    将其转换为前景（缺陷）的概率分布热力图。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 确保张量在 CPU 上并转为全精度
    logit_tensor = logit_tensor.float().detach().cpu()

    # 将 Logits 转换为概率 (经过 Softmax)
    # 形状从 [B, 2, H, W] 变成 [B, 2, H, W]
    prob_tensor = F.softmax(logit_tensor, dim=1)

    # 取出 batch=0, channel=1（假设 channel 1 是前景/缺陷类）的概率图
    # 如果你的缺陷类是 channel 0，请改成 prob_tensor[0, 0]
    prob_map = prob_tensor[0, 1].numpy()

    # 画图并保存热力图
    plt.figure(figsize=(6, 6))
    # cmap='jet' 会生成经典的红蓝热力图（红色代表概率高，蓝色代表概率低）
    plt.imshow(prob_map, cmap='jet', vmin=0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)  # 增加颜色条以方便对比数值
    plt.axis('off')
    plt.title(save_name)
    plt.savefig(os.path.join(save_dir, f"{save_name}.png"), bbox_inches='tight', dpi=150)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载模型架构...")
    model = ChangeModel(backbone="mobilenetv2").to(device)

    print("2. 正在加载训练好的权重...")
    weight_path = "/home/linweixuan/ChangeDINO/checkpoints/ESDI-10/ESDI-10_mobilenetv2_best.pth"

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

    print("3. 正在注册 LMM 模块特征提取 Hook...")
    lmm_features = {}

    # 截获进入 LMM 前的 Logits 和 经过 LMM 后的 Logits
    def hook_fn_lmm(module, input, output):
        # input[0] 是预测头 (detector) 刚出来的原始 logits
        lmm_features['1_Before_LMM-7_14'] = input[0]
        # output 是经过可学习形态学模块 (Soft Erosion/Dilation) 提纯后的 logits
        lmm_features['2_After_LMM-7_14'] = output

    # 【关键修改】：将钩子挂载到 LMM 模块 (self.refiner) 上
    hook_handle = model.refiner.register_forward_hook(hook_fn_lmm)

    print("4. 正在读取并预处理图片...")
    img_path = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test/images/7_14.jpg"

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

    print("6. 正在绘制 LMM 前后热力图...")
    save_dir = "vis_results_lmm-7_14"
    for layer_name, feat_tensor in lmm_features.items():
        print(f"   -> 保存 {layer_name} 概率图...")
        save_logit_map(feat_tensor, f"{layer_name}", save_dir=save_dir)

    # 用完后拆除钩子
    hook_handle.remove()

    print(f"\n🎉 大功告成！请去工程目录下的 {save_dir} 文件夹查看特征热力图！")

if __name__ == "__main__":
    main()