# test_model_input.py
import torch
from option import Options
from model.create_ChangeDINO import create_model

opt = Options().parse()
model = create_model(opt)

print("测试模型输入尺寸...")

# 创建测试数据
batch_size = 2
dummy_input = torch.randn(batch_size, 3, 256, 256).cuda()
dummy_label = torch.randint(0, 2, (batch_size, 256, 256)).cuda()

try:
    # 1. 测试ChangeModel
    output = model.model(dummy_input)  # model.model 是 ChangeModel
    print(f"ChangeModel输出类型: {type(output)}")
    if isinstance(output, tuple):
        print(f"  输出长度: {len(output)}")
        final_pred, preds = output  # 解包
        print(f"  final_pred形状: {final_pred.shape}")  # [2, 2, 256, 256]

        if isinstance(preds, (tuple, list)):
            print(f"  preds包含 {len(preds)} 个尺度预测")
            for i, p in enumerate(preds):
                print(f"    尺度{i}形状: {p.shape}")

    print("\n" + "=" * 50)

    # 2. 测试Model.forward（先不拆包，看看返回什么）
    print("测试Model.forward...")
    result = model(dummy_input, dummy_label)  # 这是Model类的forward

    print(f"Model.forward返回类型: {type(result)}")

    if isinstance(result, tuple):
        print(f"  返回值元组长度: {len(result)}")

        # 逐个检查返回的每个元素
        for i, item in enumerate(result):
            print(f"  元素{i}类型: {type(item)}")
            if isinstance(item, torch.Tensor):
                print(f"      形状: {item.shape}")
                print(f"      值范围: [{item.min():.3f}, {item.max():.3f}]")
            elif isinstance(item, (int, float)):
                print(f"      值: {item}")

    # 3. 尝试拆包（根据实际情况）
    if isinstance(result, tuple) and len(result) == 3:
        # 正常情况：pred, focal, dice
        pred, focal, dice = result
        if isinstance(pred, torch.Tensor):
            print(f"\n✅ 正常拆包成功!")
            print(f"  pred形状: {pred.shape}")
            print(f"  focal损失: {focal}")
            print(f"  dice损失: {dice}")
        elif isinstance(pred, tuple):
            print(f"\n⚠️  pred本身是元组，需要进一步处理")
            print(f"  pred的类型: {type(pred)}")
            print(f"  pred的长度: {len(pred) if hasattr(pred, '__len__') else 'N/A'}")

except Exception as e:
    print(f"错误: {e}")
    import traceback

    traceback.print_exc()