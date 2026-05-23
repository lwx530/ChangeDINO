import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
import torch.optim as optim
from tqdm import tqdm
import math
from util.WPFormer_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
import os
import json
import numpy as np
import random
from datetime import datetime
from util.util import make_numpy_grid, de_norm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True




class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = "val"
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = "train"

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.M = MAE()
        self.EM = Emeasure()
        self.FM = Fmeasure()  # mFβ, β²=0.3
        self.SM = Smeasure()  # Sα, α=0.5
        self.WFM = WeightedFmeasure()  # Fβw, β²=1

        self.alpha = 0.5

        self.log_path = os.path.join(self.model.save_dir, "record.txt")
        self.vis_path = os.path.join(self.model.save_dir, opt.vis_path)
        os.makedirs(self.vis_path, exist_ok=True)

        # ========== 添加TensorBoard ==========
        self.tensorboard_dir = os.path.join(self.model.save_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        print(f"📊 TensorBoard日志保存到: {self.tensorboard_dir}")
        print(f"启动命令: tensorboard --logdir={self.tensorboard_dir} --port=6006")
        # ====================================

        if not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/validation metrics\n")
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write("# time,epoch,train_loss,train_focal,train_dice,lr,")
                f.write("val_metrics(json)\n")

    def _rescheduler(self, opt):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.model.optimizer = optim.AdamW(
            trainable_params, lr=opt.lr * 0.2, weight_decay=opt.weight_decay
        )
        self.model.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.model.optimizer, int(opt.num_epochs * 0.1), eta_min=1e-7
        )
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

    def _append_log_line(self, epoch: int, train_stats: dict, val_scores: dict):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line = (
                f"{ts},{epoch},"
                f"{train_stats.get('loss', float('nan')):.6f},"
                f"{train_stats.get('focal', float('nan')):.6f},"
                f"{train_stats.get('dice', float('nan')):.6f},"
                f"{train_stats.get('lr', float('nan')):.8f},"
                + json.dumps(val_scores, ensure_ascii=False)
                + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _plot_cd_result(self, x, x2, pred, target, epoch, stage):
        """
        可视化训练/验证结果
        x: 输入图像 [B, 3, H, W]
        x2: 第二张图（单输入为None）
        pred: 预测 [B, 2, H, W] 或 [B, H, W]
        target: 标签 [B, 1, H, W] 或 [B, H, W]
        """
        # 1. 处理预测（如果是logits，取argmax）
        if pred.dim() == 4 and pred.shape[1] > 1:  # [B, 2, H, W]
            pred = torch.argmax(pred, dim=1)  # [B, H, W]

        # 2. 确保pred和target都是3维 [B, H, W]
        if pred.dim() == 4:  # [B, 1, H, W]
            pred = pred.squeeze(1)
        if target.dim() == 4:  # [B, 1, H, W]
            target = target.squeeze(1)

        # 3. 选择前8个样本可视化
        x_vis = x[0:8]
        pred_vis = pred[0:8]
        target_vis = target[0:8]

        # 4. 转换为可视化格式 [8, 3, H, W]
        # 输入图像已经是3通道
        vis_input = make_numpy_grid(de_norm(x_vis))

        # 预测和标签是单通道，复制为3通道
        pred_3ch = pred_vis.unsqueeze(1).repeat(1, 3, 1, 1)
        target_3ch = target_vis.unsqueeze(1).repeat(1, 3, 1, 1)

        vis_pred = make_numpy_grid(pred_3ch)
        vis_gt = make_numpy_grid(target_3ch)

        # 5. 拼接图像
        if x2 is not None:
            vis_input2 = make_numpy_grid(de_norm(x2[0:8]))
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        else:
            vis = np.concatenate([vis_input, vis_pred, vis_gt], axis=0)

        # 6. 保存
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_path, f"{stage}_epoch{epoch}.jpg")
        plt.imsave(file_name, vis)
        print(f"📸 可视化保存到: {file_name}")

    def train(self, epoch):
        tbar = tqdm(self.train_data, ncols=80)
        opt.phase = "train"
        _loss = 0.0
        _focal_loss = 0.0
        _dice_loss = 0.0
        last_lr = self.optimizer.param_groups[0]["lr"]

        for i, data in enumerate(tbar):
            self.model.model.train()
            pred, focal, dice = self.model(
                data["image"].cuda(), data["label"].cuda()
            )

            loss = focal * self.alpha + dice
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            last_lr = self.optimizer.param_groups[0]["lr"]
            # del loss

            # ========== 添加TensorBoard记录 ==========
            global_step = epoch * len(tbar) + i
            self.writer.add_scalar('Loss/train_total', loss.item(), global_step)
            self.writer.add_scalar('Loss/train_focal', focal.item(), global_step)
            self.writer.add_scalar('Loss/train_dice', dice.item(), global_step)
            self.writer.add_scalar('LearningRate', last_lr, global_step)
            # ========================================

            tbar.set_description(
                "Loss: %.3f, Focal: %.3f, Dice: %.3f, LR: %.6f"
                % (
                    _loss / (i + 1),
                    _focal_loss / (i + 1),
                    _dice_loss / (i + 1),
                    last_lr,
                )
            )

            if i == len(tbar) - 1:
                self._plot_cd_result(
                    data["image"], None, pred, data["label"], epoch, "train"
                )
        self.schedular.step()

        n = max(1, i + 1)
        return {
            "loss": _loss / n,
            "focal": _focal_loss / n,
            "dice": _dice_loss / n,
            "lr": last_lr,
        }

    def val(self, epoch):
        tbar = tqdm(self.val_data, ncols=80)

        # 重置所有指标
        self.M = MAE()
        self.EM = Emeasure()
        self.FM = Fmeasure()
        self.SM = Smeasure()
        self.WFM = WeightedFmeasure()

        opt.phase = "val"
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_pred = self.model.inference(
                    _data["image"].cuda()
                )
                val_target = _data["label"].detach()
                #val_pred = torch.argmax(val_pred.detach(), dim=1)

                # 1. 获取概率图（不是二值图）
                val_pred_prob = torch.softmax(val_pred.detach(), dim=1)[:, 1]  # 前景概率 [B, H, W]

                # 2. 确保标签是二维 [B, H, W]
                if val_target.dim() == 4:  # [B, 1, H, W]
                    val_target = val_target.squeeze(1)  # 变成 [B, H, W]

                # 保存处理后的标签用于可视化
                val_target_vis = val_target

                # 2. 对batch中的每个样本更新WPFormer指标
                for j in range(val_pred_prob.shape[0]):
                    # 获取单个样本
                    pred_np = val_pred_prob[j].cpu().numpy()  # [H, W]
                    target_np = val_target[j].cpu().numpy()  # [H, W]

                    # 转换为 [0, 255] uint8
                    pred_uint8 = (pred_np * 255).astype(np.uint8)
                    gt_uint8 = (target_np * 255).astype(np.uint8)

                    # 更新所有WPFormer指标
                    self.M.step(pred_uint8, gt_uint8, normalize=True)
                    self.EM.step(pred_uint8, gt_uint8, normalize=True)
                    self.FM.step(pred_uint8, gt_uint8, normalize=True)
                    self.SM.step(pred_uint8, gt_uint8, normalize=True)
                    self.WFM.step(pred_uint8, gt_uint8, normalize=True)

                if i == len(tbar) - 1:
                    val_pred_binary = (val_pred_prob > 0.5).long()
                    self._plot_cd_result(
                        _data["image"],
                        None,
                        val_pred_binary,
                        val_target_vis,
                        epoch,
                        "val",
                    )
            M_result = self.M.get_results()
            EM_result = self.EM.get_results()
            FM_result = self.FM.get_results()
            SM_result = self.SM.get_results()
            WFM_result = self.WFM.get_results()

            val_scores = {
                'MAE': M_result['mae'],
                'meanEm': EM_result['em']['adp'],
                'Fmeasure': FM_result['fm']['adp'],
                'Smeasure': SM_result['sm'],
                'wFmeasure': WFM_result['wfm']
            }

            # ========== 添加TensorBoard记录 ==========
            for k, v in val_scores.items():
                self.writer.add_scalar(f'Metrics/val_{k}', v, epoch)

            # 特别记录最佳IoU
            current_best = val_scores.get('wFmeasure', 0)
            if current_best >= self.previous_best:
                self.writer.add_scalar('Metrics/best_wFmeasure', current_best, epoch)
            # ========================================

            message = "(phase: %s) " % (self.opt.phase)
            for k, v in val_scores.items():
                message += "%s: %.6f " % (k, v)
            print(message)

        current_score = val_scores.get('wFmeasure', 0.0)
        if current_score >= self.previous_best:
            self.model.save(self.opt.name, self.opt.backbone)
            self.previous_best = current_score

        return val_scores


if __name__ == "__main__":
    opt = Options().parse()
    trainval = Trainval(opt)
    setup_seed(seed=1)

    try:
        for epoch in range(1, opt.num_epochs + 1):
            print(
                "\n==> Name %s, Epoch %i, previous best = %.6f"
                % (opt.name, epoch, trainval.previous_best)
            )
            if epoch == int(opt.num_epochs * 0.9):
                trainval._rescheduler(opt)
            train_stats = trainval.train(epoch)
            val_scores = trainval.val(epoch)

            trainval._append_log_line(epoch, train_stats, val_scores)
    finally:
        # ========== 关闭TensorBoard writer ==========
        if hasattr(trainval, 'writer'):
            trainval.writer.close()
        # ============================================

    print("Done!")
