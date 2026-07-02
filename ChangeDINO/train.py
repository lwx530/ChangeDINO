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

        self.opt.phase = "test"
        test_loader = DataLoader(opt)
        self.test_data = test_loader.load_data()
        test_size = len(test_loader)
        print("#testing images = %d" % test_size)
        self.opt.phase = "train"

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.M = MAE()
        self.EM = Emeasure()
        self.FM = Fmeasure()
        self.SM = Smeasure()
        self.WFM = WeightedFmeasure()

        self.alpha = 0.5
        self.beta = 0.5

        self.log_path = os.path.join(self.model.save_dir, "record.txt")
        self.vis_path = os.path.join(self.model.save_dir, opt.vis_path)
        os.makedirs(self.vis_path, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/testing metrics\n")  # 修改了文字说明
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write("# time,epoch,train_loss,train_focal,train_dice,lr,")
                f.write("test_metrics(json)\n")  # 修改了文字说明

    def _rescheduler(self, opt):
        self.model.optimizer = optim.AdamW(
            self.model.model.parameters(), lr=opt.lr * 0.2, weight_decay=opt.weight_decay
        )
        self.model.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.model.optimizer, int(opt.num_epochs * 0.1), eta_min=1e-7
        )
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

    def _append_log_line(self, epoch: int, train_stats: dict, test_scores: dict):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line = (
                f"{ts},{epoch},"
                f"{train_stats.get('loss', float('nan')):.6f},"
                f"{train_stats.get('hybrid', float('nan')):.6f},"
                f"{train_stats.get('lr', float('nan')):.8f},"
                + json.dumps(test_scores, ensure_ascii=False)
                + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def train(self, epoch):
        tbar = tqdm(self.train_data, ncols=80)
        self.opt.phase = "train"
        _loss = 0.0
        _hybrid = 0.0
        _boundary_loss = 0.0
        last_lr = self.optimizer.param_groups[0]["lr"]

        for i, data in enumerate(tbar):
            self.model.model.train()
            pred, hybrid, boundary = self.model(
                data["image"].cuda(), data["label"].cuda()
            )

            loss = hybrid + boundary * self.beta
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _loss += loss.item()
            _hybrid += hybrid.item()
            _boundary_loss += boundary.item()
            last_lr = self.optimizer.param_groups[0]["lr"]

            tbar.set_description(
                "Loss: %.3f, Hybrid: %.3f, Bnd: %.3f"
                % (
                    _loss / (i + 1),
                    _hybrid / (i + 1),
                    _boundary_loss / (i + 1)
                )
            )

        self.schedular.step()

        n = max(1, i + 1)
        return {
            "loss": _loss / n,
            "hybrid": _hybrid / n,
            "lr": last_lr,
        }

    def test(self, epoch):
        tbar = tqdm(self.test_data, ncols=80)

        # 重置所有指标
        self.M = MAE()
        self.EM = Emeasure()
        self.FM = Fmeasure()
        self.SM = Smeasure()
        self.WFM = WeightedFmeasure()

        self.opt.phase = "test"
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_pred = self.model.inference(
                    _data["image"].cuda()
                )
                val_target = _data["label"].detach()

                # 获取概率图
                val_pred_prob = torch.softmax(val_pred.detach(), dim=1)[:, 1]

                # 确保标签是二维
                if val_target.dim() == 4:
                    val_target = val_target.squeeze(1)

                    # 对batch中的每个样本更新WPFormer指标
                for j in range(val_pred_prob.shape[0]):
                    pred_np = val_pred_prob[j].cpu().numpy()
                    target_np = val_target[j].cpu().numpy()

                    pred_uint8 = (pred_np * 255).astype(np.uint8)
                    gt_uint8 = (target_np * 255).astype(np.uint8)

                    self.M.step(pred_uint8, gt_uint8, normalize=True)
                    self.EM.step(pred_uint8, gt_uint8, normalize=True)
                    self.FM.step(pred_uint8, gt_uint8, normalize=True)
                    self.SM.step(pred_uint8, gt_uint8, normalize=True)
                    self.WFM.step(pred_uint8, gt_uint8, normalize=True)

            M_result = self.M.get_results()
            EM_result = self.EM.get_results()
            FM_result = self.FM.get_results()
            SM_result = self.SM.get_results()
            WFM_result = self.WFM.get_results()

            test_scores = {
                'MAE': M_result['mae'],
                'meanEm': EM_result['em']['adp'],
                'Fmeasure': FM_result['fm']['adp'],
                'Smeasure': SM_result['sm'],
                'wFmeasure': WFM_result['wfm']
            }

            message = "(phase: %s) " % (self.opt.phase)
            for k, v in test_scores.items():
                message += "%s: %.6f " % (k, v)
            print(message)

        # 这里你用的是 wFmeasure 作为最佳模型的判断标准，也可以换成和 WPFormer 一样的 Smeasure
        current_score = test_scores.get('Smeasure', 0.0)
        if current_score >= self.previous_best:
            self.model.save(self.opt.name, self.opt.backbone)
            self.previous_best = current_score

        return test_scores


if __name__ == "__main__":
    opt = Options().parse()
    trainval = Trainval(opt)
    setup_seed(seed=1)

    epoch_val = 60

    try:
        for epoch in range(1, opt.num_epochs + 1):
            print(
                "\n==> Name %s, Epoch %i, previous best = %.6f"
                % (opt.name, epoch, trainval.previous_best)
            )
            if epoch == int(opt.num_epochs * 0.9):
                trainval._rescheduler(opt)

            # 训练阶段
            train_stats = trainval.train(epoch)

            # 评估阶段 (只有大于等于 epoch_val 才进行测试集推理计算)
            if epoch >= epoch_val:
                test_scores = trainval.test(epoch)
                trainval._append_log_line(epoch, train_stats, test_scores)
            else:
                trainval._append_log_line(epoch, train_stats, {"Message": "Skipped test"})

    finally:
        pass

    print("Done!")