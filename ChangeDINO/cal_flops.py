import torch
from thop import profile
from model.ChangeDINO import ChangeModel # 导入你的模型

def check_params_flops():
    # 实例化模型 (根据你目前的设定)
    model = ChangeModel(backbone="mobilenetv2", fpn_channels=128).cuda()
    model.eval()

    # 构造一个虚拟输入张量 (请改成你真实的训练/测试尺寸，比如 384x384 或 256x256)
    # 尺寸不同，FLOPs(计算量) 会差很多！
    dummy_input = torch.randn(1, 3, 256, 256).cuda()

    print("⏳ 正在计算 FLOPs 和 Params，请稍候...")
    flops, params = profile(model, inputs=(dummy_input, ))

    print(f"✅ 计算完成！输入尺寸为 {dummy_input.shape[2:]} 时：")
    print(f"🔸 FLOPs (计算量): {flops / 1e9:.2f} G")
    print(f"🔸 Params (总参数量): {params / 1e6:.2f} M")

if __name__ == "__main__":
    check_params_flops()