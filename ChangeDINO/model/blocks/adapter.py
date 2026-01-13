import torch
import torch.nn as nn
import torch.nn.functional as F
import re

REPO_DIR = "dinov3"
DINO_NAME = "dinov3_vitl16"
MODEL_TO_NUM_LAYERS = {
    "VITS": 12,
    "VITSP": 12,
    "VITB": 12,
    "VITL": 24,
    "VITHP": 32,
    "VIT7B": 40,
}


class DINOV3Wrapper(nn.Module):
    def __init__(
        self,
        # weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        extract_ids=[5, 11, 17, 23],
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.model = torch.hub.load(
            REPO_DIR,
            DINO_NAME,
            source="local",
            weights=weights_path,
        )
        self.model = self.model.eval().to(device)
        self.n_layers = MODEL_TO_NUM_LAYERS[
            re.sub(r"\d+", "", DINO_NAME.split("_")[-1]).upper()
        ]
        self.patch_size = int(re.findall(r"\d+", DINO_NAME.split("_")[-1])[-1])
        self.extract_ids = extract_ids

        # freeze the backbone
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    x, n=range(self.n_layers), reshape=True, norm=True
                )
                feats_ = []
                for i in range(len(self.extract_ids)):
                    feats_.append(feats[self.extract_ids[i]])  # [B, N, C]
        return feats_


class SepAdapterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 64, act=nn.SiLU):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                r, r, kernel_size=3, padding=1, groups=r, bias=False
            ),  # depthwise
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return x


class DenseAdapterLite(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        sizes=(64, 32, 16, 8),
        bottleneck=64,
        share=False,
        # 新增参数
        use_attention=True,
        keep_resolution=True,  # 是否保持原始分辨率
    ):
        super().__init__()
        self.sizes = list(sizes)
        self.keep_resolution = keep_resolution

        if share:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SepAdapterBlock(in_dim, out_dim, r=bottleneck) for _ in self.sizes]
            )

        # 可选的注意力增强
        if use_attention:
            self.attentions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_dim, out_dim // 4, 1),
                    nn.ReLU(),
                    nn.Conv2d(out_dim // 4, 1, 1),
                    nn.Sigmoid()
                ) for _ in self.sizes
            ])
        else:
            self.attentions = None

        self.share = share

    def forward(self, feats):
        """
        feats: list of 4 tensors, each [B, C, H_i, W_i]（C = in_dim）
        return: list of 4 tensors, each [B, out_dim, S_i, S_i], S_i ∈ self.sizes
        """
        outs = []
        for i, x in enumerate(feats):
            # 如果保持分辨率，使用原始尺寸
            if self.keep_resolution:
                target_size = x.shape[-2:]
            else:
                target_size = (self.sizes[i], self.sizes[i])

            # 调整尺寸
            if x.shape[-2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            out = block(x)

            # 注意力增强（突出重要区域）
            if self.attentions is not None:
                attn = self.attentions[i](out)
                out = out * attn + out  # 残差注意力

            outs.append(out)
        return outs
