from .ChangeDINO import ChangeModel
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import os
import torch.optim as optim
from .loss.focal import FocalLoss
from .loss.dice import DICELoss
from .loss.boundary import BoundaryLoss
from .loss.hybrid_loss import HybridLoss

def get_model(backbone_name="mobilenetv2", fpn_channels=128, n_layers=[1, 1, 1], **kwargs):
    model = ChangeModel(backbone_name, fpn_channels, n_layers=n_layers, **kwargs)
    # print(model)
    return model


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device(
            "cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu"
        )
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = get_model(
            backbone_name=opt.backbone,
            fpn_name=opt.fpn,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            n_layers=opt.n_layers,
            extract_ids=opt.extract_ids,
        )
        # self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        # self.dice = DICELoss()
        self.hybrid_loss = HybridLoss()
        self.boundary_loss = BoundaryLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
        )

        self.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, opt.num_epochs, eta_min=1e-7
        )
        if opt.load_pretrain:
            self.load_ckpt(self.model, self.optimizer, opt.name, opt.backbone)
        self.model.cuda()

        print("---------- Networks initialized -------------")

    def forward(self, x, label):
        final_pred, preds, edge_mask = self.model(x)
        # label = label.long()
        hybrid = self.hybrid_loss(final_pred, label)
        for p in preds:
            hybrid += 0.5 * self.hybrid_loss(p, label)

        edge_mask_up = F.interpolate(
            edge_mask,
            size=label.shape[-2:],  # 获取 label 的 H, W
            mode="bilinear",
            align_corners=False
        )
        boundary = self.boundary_loss(edge_mask_up, label)

        return final_pred, hybrid, boundary

    @torch.inference_mode()
    def inference(self, x):
        pred = self.model._forward(x)
        return pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = "%s_%s_best.pth" % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            checkpoint = torch.load(
                save_path, map_location=self.device, weights_only=True
            )
            network.load_state_dict(checkpoint["network"], strict=False)
            print("load pre-trained")

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = "%s_%s_best.pth" % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(
            {
                "network": network.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )
        if torch.cuda.is_available():
            network.cuda()

    def save(self, model_name, backbone):
        self.save_ckpt(self.model, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()
