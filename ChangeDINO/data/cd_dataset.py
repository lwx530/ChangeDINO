from .transform import Transforms
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def make_dataset(root):
    img_paths = []
    names = []

    assert os.path.isdir(root), "%s is not a valid directory" % root

    for fname in sorted(os.listdir(root)):
        path = os.path.join(root, fname)
        if os.path.isfile(path):
            img_paths.append(path)
            names.append(fname)

    return img_paths, names

class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt
        self.is_train = (opt.phase == "train")

        # self.image_resize = transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)
        # self.label_resize = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)

        self.image_resize = transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)
        self.label_resize = transforms.Resize((384, 384), interpolation=InterpolationMode.NEAREST)

        self.image_dir = os.path.join(opt.dataroot, opt.dataset, opt.phase, "images")
        self.image_paths, self.fnames = make_dataset(self.image_dir)

        self.label_dir = os.path.join(opt.dataroot, opt.dataset, opt.phase, "gt")

        # 创建对应的标签文件路径
        self.label_paths = []
        for fname in self.fnames:
            base_name = os.path.splitext(fname)[0]
            label_path = os.path.join(self.label_dir, base_name + ".png")

            if not os.path.exists(label_path):
                print(f"警告: 找不到对应的标签文件: {fname}")

            self.label_paths.append(label_path)

        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
        self.transform = Transforms()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # 加载单张图像
        img_path = self.image_paths[index]
        fname = self.fnames[index]
        img = Image.open(img_path).convert('RGB')  # 确保RGB

        # 加载标签
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')  # 灰度图

        # 数据增强（需要调整Transform类）
        if self.is_train:
            _data = self.transform({"image": img, "label": label})
            img = _data["image"]
            label = _data["label"]

        img = self.image_resize(img)
        label = self.label_resize(label)

        img_tensor = self.to_tensor(img)
        img_tensor = self.normalize(img_tensor)

        # 标签转换：PIL Image -> numpy -> 二值化 -> Tensor
        label_np = np.array(label)
        if label_np.max() > 1:
            label_np = label_np / 255.0
        label_np = (label_np > 0.5).astype(np.float32)

        label_tensor = torch.from_numpy(label_np).float().unsqueeze(0)  # [1, H, W]

        input_dict = {"image": img_tensor, "label": label_tensor, "fname": fname, "label_path": label_path}

        return input_dict


class DataLoader:

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
