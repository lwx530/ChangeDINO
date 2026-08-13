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
        pred1, edge_mask = self.model(x)
        label = label.long()
        loss1 = self.hybrid_loss(pred1, label)
        # loss2 = self.hybrid_loss(pred2, label)
        # loss3 = self.hybrid_loss(pred3, label)
        # loss4 = self.hybrid_loss(pred4, label)

        edge_mask_up = F.interpolate(
            edge_mask,
            size=label.shape[-2:],  # 获取 label 的 H, W
            mode="bilinear",
            align_corners=False
        )
        boundary = self.boundary_loss(edge_mask_up, label)

        hybrid = loss1

        loss = hybrid + 0.2 * boundary

        return pred1, loss

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

    def save_latest(self, epoch, previous_best):
        save_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save({
            'epoch': epoch,
            'previous_best': previous_best,
            'network': self.model.cpu().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.schedular.state_dict(),
        }, save_path)
        if torch.cuda.is_available():
            self.model.cuda()

    def resume_latest(self):
        save_path = os.path.join(self.save_dir, 'latest.pth')
        if not os.path.isfile(save_path):
            print('No latest checkpoint found, starting from scratch')
            return 1, 0.0
        checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedular.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        previous_best = checkpoint.get('previous_best', 0.0)
        print('Resumed from epoch %d, previous best = %.6f' % (checkpoint['epoch'], previous_best))
        return start_epoch, previous_best

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()
