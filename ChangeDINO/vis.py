import torch
from PIL import Image
import torchvision.transforms as transforms
import os

# 导入你的模型
from model.ChangeDINO import ChangeModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. 正在加载模型架构...")
    # ========================================================
    # 【请确认】这里使用你目前效果最好的 backbone (比如 mobilenetv2)
    # ========================================================
    model = ChangeModel(backbone="resnet18d").to(device)

    print("2. 正在加载训练好的权重...")
    # ========================================================
    # 【非常重要】填入你训练出来的最佳权重的相对或绝对路径！
    # 如果不加载权重，画出来的图将是毫无意义的雪花噪点。
    # ========================================================
    weight_path = "/home/linweixuan/ChangeDINO/checkpoints/ESDI/ESDI_resnet18d_best.pth"  # <--- 请修改这里！

    if os.path.exists(weight_path):
        # 你的保存格式可能包含在外层字典中，比如 ['model_state_dict'] 或直接是模型权重
        checkpoint = torch.load(weight_path, map_location=device)
        # 根据你实际的保存格式修改键名，常见的有 'model_state_dict', 'state_dict' 或直接加载
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功！")
    else:
        print(f"❌ 警告：未找到权重文件 {weight_path}，将使用随机权重进行可视化（不推荐）")

    model.eval()  # 务必切到测试模式，关闭 Dropout 和 BatchNorm 的更新

    print("3. 正在读取并预处理图片...")
    # ========================================================
    # 【请修改】填上一张你精心挑选的、带有明显缺陷的验证集图片路径
    # ========================================================
    img_path = "/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD/test/images/7_12.jpg"  # <--- 请修改这里！

    if not os.path.exists(img_path):
        print(f"❌ 找不到图片：{img_path}")
        return

    img = Image.open(img_path).convert('RGB')

    # 【核心对齐】严格复刻你 cd_dataset.py 中测试阶段的预处理逻辑
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 你的 dataset 中是 256x256
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)  # 变成 [B=1, C=3, H=256, W=256]

    print("4. 正在进行前向推理并提取特征图...")
    # 这里触发前向传播。只要你 ChangeDINO.py 里面 Encoder 的 VISUALIZE 设为了 True
    # 它运行到对应的行就会自动把热力图保存到 vis_results 文件夹
    with torch.no_grad():
        # 调用推理接口，这内部会调用 self.encoder(x) 触发我们埋点的画图代码
        model._forward(input_tensor)

    print("\n🎉 大功告成！请去工程目录下的 vis_results 文件夹查看特征热力图！")


if __name__ == "__main__":
    main()