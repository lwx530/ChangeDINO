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

def get_model(**kwargs):
    model = ChangeModel(**kwargs)
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

    def forward(self, x, label):
        pr5, pr6, pr7, pr8, p1, p2, p3, p4, edge1, edge2 = self.model(x)

        lr1 = self.hybrid_loss(pr5, label)
        lr2 = self.hybrid_loss(pr6, label)
        lr3 = self.hybrid_loss(pr7, label)
        lr4 = self.hybrid_loss(pr8, label)
        l1 = self.hybrid_loss(p1, label)
        l2 = self.hybrid_loss(p2, label)
        l3 = self.hybrid_loss(p3, label)
        l4 = self.hybrid_loss(p4, label)

        edge1_mask_up = F.interpolate(
            edge1,
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        lb1 = self.boundary_loss(edge1_mask_up, label) * 0.5

        edge2_mask_up = F.interpolate(
            edge2,
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        lb2 = self.boundary_loss(edge1_mask_up, label) * 0.5

        loss = lr1 + lr2 + lr3 + lr4 + l1 + l2 + l3 + l4 + lb1 + lb2

        return pr5, loss

    @torch.inference_mode()
    def inference(self, x):
        return self.model._forward(x)

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
