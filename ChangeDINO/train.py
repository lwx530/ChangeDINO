import os
import json
import random
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
from util.WPFormer_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_log_line(log_path, epoch, train_stats, scores):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts},{epoch},"
        f"{train_stats.get('loss', float('nan')):.6f},"
        f"{train_stats.get('lr', float('nan')):.8f},"
        + json.dumps(scores, ensure_ascii=False)
        + "\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)

def evaluate(net, data_loader, verbose=True):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    net.eval()
    with torch.no_grad():
        for _data in tqdm(data_loader, ncols=80, disable=not verbose):
            logits = net.inference(_data["image"].cuda())

            if logits.shape[1] == 2:
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                prob = torch.sigmoid(logits[:, 0]).cpu().numpy()

            for j in range(prob.shape[0]):
                p = prob[j]
                p = (p - p.min()) / (p.max() - p.min() + 1e-8)

                img = Image.fromarray((p * 255).astype(np.float32)).convert("L")

                gt = cv2.imread(_data["label_path"][j], cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    raise FileNotFoundError(_data["label_path"][j])
                H, W = gt.shape
                img = img.resize((W, H), resample=Image.NEAREST)

                pred = np.array(img)
                FM.step(pred=pred, gt=gt)
                WFM.step(pred=pred, gt=gt)
                SM.step(pred=pred, gt=gt)
                EM.step(pred=pred, gt=gt)
                M.step(pred=pred, gt=gt)

    return {
        "MAE":       M.get_results()["mae"],
        "meanEm":    EM.get_results()["em"]["curve"].mean(),
        "meanFm":    FM.get_results()["fm"]["curve"].mean(),
        "Smeasure":  SM.get_results()["sm"],
        "wFmeasure": WFM.get_results()["wfm"],
    }


def train(opt):
    train_loader = DataLoader(opt)
    train_data = train_loader.load_data()
    print("#training images = %d" % len(train_loader))

    opt.phase = "test"
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    print("#testing images = %d" % len(test_loader))
    opt.phase = "train"

    net = create_model(opt)
    optimizer = net.optimizer
    schedular = net.schedular

    assert opt.epoch_val <= opt.num_epochs, \
        "epoch_val(%d) 不能大于 num_epochs(%d)" % (opt.epoch_val, opt.num_epochs)

    log_path = os.path.join(net.save_dir, "record.txt")
    if not os.path.exists(log_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("# Record of training metrics\n")
            f.write("# name: %s | backbone: %s\n" % (opt.name, opt.backbone))
            f.write("# time,epoch,train_loss,lr,metrics(json)\n")

    print("--- start training ---")
    previous_best = 0.0

    for epoch in range(1, opt.num_epochs + 1):
        print("\n==> Name %s, Epoch %i, previous best = %.6f"
              % (opt.name, epoch, previous_best))

        tbar = tqdm(train_data, ncols=80)
        net.train()
        _loss = 0.0
        last_lr = optimizer.param_groups[0]["lr"]
        i = 0
        for i, data in enumerate(tbar):
            _, loss = net(data["image"].cuda(), data["label"].cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _loss += loss.item()
            last_lr = optimizer.param_groups[0]["lr"]
            tbar.set_description("Loss: %.3f" % (_loss / (i + 1)))

        schedular.step()

        train_stats = {"loss": _loss / max(1, i + 1), "lr": last_lr}

        if epoch >= opt.epoch_val:
            scores = evaluate(net, test_data, verbose=False)
            print("(epoch %d) " % epoch
                  + " ".join("%s: %.6f" % (k, v) for k, v in scores.items()))

            current_score = scores["wFmeasure"]
            if current_score >= previous_best:
                net.save(opt.name, opt.backbone)
                previous_best = current_score

            append_log_line(log_path, epoch, train_stats, scores)
        else:
            append_log_line(log_path, epoch, train_stats, {"Message": "Skipped eval"})

    print("Done!")


if __name__ == "__main__":
    opt = Options().parse()
    setup_seed(seed=opt.seed)
    train(opt)