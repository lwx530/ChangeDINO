import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class Transforms(object):
    def __call__(self, _data):
        image, label = _data['image'], _data['label']

        # 2. 水平翻转（对image和label同时应用）
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # 3. 垂直翻转（对image和label同时应用）
        if random.random() < 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # 4. 旋转（对image和label同时应用）
        if random.random() < 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            label = TF.rotate(label, angle, interpolation=InterpolationMode.NEAREST)

        # 5. 颜色增强（只对图像，不对标签）
        if random.random() < 0.5:
            colorjitters = []
            brightness_factor = random.uniform(0.75, 1.25)
            colorjitters.append(Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))
            contrast_factor = random.uniform(0.75, 1.25)
            colorjitters.append(Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))
            saturation_factor = random.uniform(0.75, 1.25)
            colorjitters.append(Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))
            random.shuffle(colorjitters)
            colorjitter = Compose(colorjitters)
            image = colorjitter(image)
            # 注意：label不进行颜色增强

        # 6. 随机裁剪和缩放（对image和label同时应用）

        i, j, h, w = transforms.RandomResizedCrop(size=(256, 256)).get_params(
            img=image, scale=[0.333, 1.0], ratio=[0.75, 1.333]
        )
        image = TF.resized_crop(image, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.BILINEAR)
        label = TF.resized_crop(label, i, j, h, w, size=(256, 256), interpolation=InterpolationMode.NEAREST)

        return {'image': image, 'label': label}

class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string





