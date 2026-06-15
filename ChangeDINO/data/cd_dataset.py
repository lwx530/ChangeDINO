from .transform import Transforms
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def make_dataset(dir):
    img_paths = []
    names = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            img_paths.append(path)
            names.append(fname)

    return img_paths, names


class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt

        self.resize = transforms.Resize((384, 384))

        '''self.dir1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, "A")
        self.t1_paths, self.fnames = sorted(make_dataset(self.dir1))

        self.dir2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, "B")
        self.t2_paths, _ = sorted(make_dataset(self.dir2))

        self.dir_label = os.path.join(opt.dataroot, opt.dataset, opt.phase, "label")
        self.label_paths, _ = sorted(make_dataset(self.dir_label))'''

        # 单输入分割 - 图像是jpg，标签是png
        self.image_dir = os.path.join(opt.dataroot, opt.dataset, opt.phase, "images")
        self.image_paths, self.fnames = sorted(make_dataset(self.image_dir))

        self.label_dir = os.path.join(opt.dataroot, opt.dataset, opt.phase, "gt")

        # 创建对应的标签文件路径
        self.label_paths = []
        for fname in self.fnames:
            # jpg -> png 转换
            # 例如: image_001.jpg -> image_001.png
            label_name = fname.replace('.jpg', '.png').replace('.JPG', '.png')
            label_path = os.path.join(self.label_dir, label_name)

            # 如果直接替换不行，尝试其他命名规则
            if not os.path.exists(label_path):
                # 方案1: 去掉.jpg后缀再加.png
                label_name = fname.split('.')[0] + '.png'
                label_path = os.path.join(self.label_dir, label_name)

            if not os.path.exists(label_path):
                # 方案2: 可能存在不同的前缀/后缀
                # 例如: train_001.jpg -> gt_train_001.png
                print(f"警告: 找不到对应的标签文件: {fname}")

            self.label_paths.append(label_path)

        self.dataset_size = len(self.image_paths)

        self.normalize = transforms.Compose(
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
        self.transform = transforms.Compose([Transforms()])
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        '''t1_path = self.t1_paths[index]
        fname = self.fnames[index]
        img1 = Image.open(t1_path)

        t2_path = self.t2_paths[index]
        img2 = Image.open(t2_path)

        label_path = self.label_paths[index]
        label = np.array(Image.open(label_path)) / 255
        label[label > 0] = 1
        cd_label = Image.fromarray(label)

        if self.opt.phase == "train":
            _data = self.transform({"img1": img1, "img2": img2, "cd_label": cd_label})
            img1, img2, cd_label = _data["img1"], _data["img2"], _data["cd_label"]

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        cd_label = torch.from_numpy(np.array(cd_label))
        input_dict = {"img1": img1, "img2": img2, "cd_label": cd_label, "fname": fname}'''

        # 加载单张图像
        img_path = self.image_paths[index]
        fname = self.fnames[index]
        img = Image.open(img_path).convert('RGB')  # 确保RGB

        # 加载标签
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')  # 灰度图

        # 数据增强（需要调整Transform类）
        if self.opt.phase == "train":
            _data = self.transform({"image": img, "label": label})
            img = _data["image"]
            label = _data["label"]
        else:
            # 验证/测试时：直接resize到256×256
            img = self.resize(img)
            label = self.resize(label)

        img_tensor = self.to_tensor(img)
        img_tensor = self.normalize(img_tensor)

        # 标签转换：PIL Image -> numpy -> 二值化 -> Tensor
        label_np = np.array(label)
        if label_np.max() > 1:
            label_np = label_np / 255.0
        label_np = (label_np > 0.5).astype(np.float32)

        label_tensor = torch.from_numpy(label_np).float().unsqueeze(0)  # [1, H, W]

        input_dict = {"image": img_tensor, "label": label_tensor, "fname": fname}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.phase == "train",
            pin_memory=True,
            drop_last=opt.phase == "train",
            num_workers=int(opt.num_workers),
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
