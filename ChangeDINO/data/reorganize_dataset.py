import os
import shutil
import random
from pathlib import Path


def split_train_val(dataset_root="/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD",
                    train_ratio=0.85,
                    seed=42):
    """
    将训练集划分为train和val两个子集
    """
    print("=" * 50)
    print("🚀 开始划分训练集为train和val")
    print("=" * 50)

    # 设置随机种子
    random.seed(seed)

    # 原始训练集路径
    train_path = os.path.join(dataset_root, "train")

    if not os.path.exists(train_path):
        print(f"❌ 训练集路径不存在: {train_path}")
        return

    # 创建val文件夹
    val_path = os.path.join(dataset_root, "val")
    if os.path.exists(val_path):
        print(f"⚠️  val文件夹已存在，将清空后重新创建")
        shutil.rmtree(val_path)

    # 创建val文件夹结构
    os.makedirs(os.path.join(val_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_path, "gt"), exist_ok=True)

    # 获取所有图像文件（假设图像是jpg，标签是png）
    images_dir = os.path.join(train_path, "images")
    gt_dir = os.path.join(train_path, "gt")

    # 获取所有图像文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))

    image_files = [str(f) for f in image_files]

    print(f"📊 原始训练集统计:")
    print(f"  - 总图像数: {len(image_files)}")

    # 随机打乱
    random.shuffle(image_files)

    # 划分
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    print(f"📊 划分后统计:")
    print(f"  - 新训练集: {len(train_images)} 张 ({train_ratio * 100:.1f}%)")
    print(f"  - 验证集: {len(val_images)} 张 ({(1 - train_ratio) * 100:.1f}%)")

    # 创建新训练集文件夹（备份原始）
    train_backup_path = os.path.join(dataset_root, "train_backup")
    if not os.path.exists(train_backup_path):
        print(f"📁 备份原始训练集到: {train_backup_path}")
        shutil.copytree(train_path, train_backup_path)

    # 移动验证集文件
    print(f"📦 移动验证集文件到val文件夹...")
    moved_count = 0

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        img_stem = Path(img_name).stem

        # 移动图像文件
        dest_img_path = os.path.join(val_path, "images", img_name)
        shutil.move(img_path, dest_img_path)

        # 移动对应的标签文件
        # 尝试多种可能的标签文件名
        possible_gt_names = [
            f"{img_stem}.png",
            f"{img_stem}.PNG",
            f"{img_stem}.jpg".replace('.jpg', '.png'),  # 如果图像是jpg
            f"{img_stem}.jpeg".replace('.jpeg', '.png')
        ]

        gt_found = False
        for gt_name in possible_gt_names:
            src_gt_path = os.path.join(gt_dir, gt_name)
            if os.path.exists(src_gt_path):
                dest_gt_path = os.path.join(val_path, "gt", gt_name)
                shutil.move(src_gt_path, dest_gt_path)
                gt_found = True
                break

        if not gt_found:
            print(f"  ⚠️  找不到图像 {img_name} 对应的标签文件")

        moved_count += 1
        if moved_count % 100 == 0:
            print(f"  已移动 {moved_count}/{len(val_images)} 个样本")

    # 检查结果
    print("\n✅ 划分完成！")
    print("=" * 50)

    # 验证划分结果
    train_img_count = len(list(Path(os.path.join(train_path, "images")).glob("*")))
    train_gt_count = len(list(Path(os.path.join(train_path, "gt")).glob("*")))
    val_img_count = len(list(Path(os.path.join(val_path, "images")).glob("*")))
    val_gt_count = len(list(Path(os.path.join(val_path, "gt")).glob("*")))

    print("📊 最终统计:")
    print(f"  新训练集:")
    print(f"    - 图像: {train_img_count} 张")
    print(f"    - 标签: {train_gt_count} 张")
    print(f"  验证集:")
    print(f"    - 图像: {val_img_count} 张")
    print(f"    - 标签: {val_gt_count} 张")

    # 检查对应关系
    if train_img_count == train_gt_count:
        print(f"  ✅ 新训练集图像-标签匹配")
    else:
        print(f"  ⚠️  新训练集图像-标签不匹配")

    if val_img_count == val_gt_count:
        print(f"  ✅ 验证集图像-标签匹配")
    else:
        print(f"  ⚠️  验证集图像-标签不匹配")

    print(f"\n💾 原始训练集已备份到: {train_backup_path}")
    print("=" * 50)


'''def create_data_list(dataset_root="/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD"):
    """
    创建数据列表文件（可选）
    """
    for phase in ['train', 'val', 'test']:
        phase_path = os.path.join(dataset_root, phase)
        if not os.path.exists(phase_path):
            continue

        images_dir = os.path.join(phase_path, "images")
        gt_dir = os.path.join(phase_path, "gt")

        if not os.path.exists(images_dir) or not os.path.exists(gt_dir):
            continue

        # 创建列表文件
        list_file = os.path.join(dataset_root, f"{phase}_list.txt")
        with open(list_file, 'w') as f:
            # 获取所有图像文件
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                for img_path in Path(images_dir).glob(f"*{ext}"):
                    img_name = img_path.name
                    img_stem = img_path.stem

                    # 查找对应的标签文件
                    gt_found = False
                    for gt_ext in ['.png', '.PNG']:
                        gt_path = os.path.join(gt_dir, f"{img_stem}{gt_ext}")
                        if os.path.exists(gt_path):
                            f.write(f"{img_name}\t{os.path.basename(gt_path)}\n")
                            gt_found = True
                            break

                    if not gt_found:
                        print(f"警告: {img_name} 没有对应的标签文件")

        print(f"📝 创建列表文件: {list_file}")'''


if __name__ == "__main__":
    # 1. 划分数据集
    print("🔧 步骤1: 划分训练集为train和val")
    split_train_val(
        dataset_root="/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD",
        train_ratio=0.8,  # 80%训练，20%验证
        seed=42
    )

    print("\n" + "=" * 50)

    ''''# 2. 创建数据列表文件（可选）
    print("🔧 步骤2: 创建数据列表文件")
    create_data_list("/home/linweixuan/ChangeDINO/datasets/ESDIs-SOD")

    print("\n✅ 所有操作完成！")'''