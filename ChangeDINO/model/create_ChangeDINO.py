from .ChangeDINO import ChangeModel
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import os
import torch.optim as optim
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
            backbone=opt.backbone,
            fpn_channels=opt.fpn_channels,
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
        pred_s, coarse_masks = self.model(x)
        label = label.long()

        # 1. 最终输出损失
        loss_main = self.hybrid_loss(pred_s, label)

        # 2. GLF 各级原型粗预测深监督 (逐层回传梯度)
        loss_aux = 0.0
        aux_weights = [0.4, 0.3, 0.2, 0.1]
        for s_mask, w in zip(coarse_masks, aux_weights):
            if s_mask.shape[-2:] != label.shape[-2:]:
                s_mask = F.interpolate(s_mask, size=label.shape[-2:], mode="bilinear", align_corners=False)
            loss_aux += w * self.hybrid_loss(s_mask, label)

        total_loss = loss_main + loss_aux
        return pred_s, total_loss

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
