# import os
# import json
# import random
# from datetime import datetime
#
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from tqdm import tqdm
# import torch.nn.functional as F
#
# from option import Options
# from data.cd_dataset import DataLoader
# from model.create_ChangeDINO import create_model
# from util.SOD_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
#
#
# def setup_seed(seed):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# def append_log_line(log_path, epoch, train_stats, scores):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     line = (
#         f"{ts},{epoch},"
#         f"{train_stats.get('loss', float('nan')):.6f},"
#         f"{train_stats.get('lr', float('nan')):.8f},"
#         + json.dumps(scores, ensure_ascii=False)
#         + "\n"
#     )
#     with open(log_path, "a", encoding="utf-8") as f:
#         f.write(line)
#
# def evaluate(model, data_loader, save_dir=None):
#     FM = Fmeasure()
#     WFM = WeightedFmeasure()
#     SM = Smeasure()
#     EM = Emeasure()
#     M = MAE()
#
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#
#     model.eval()
#     with torch.no_grad():
#         for _data in tqdm(data_loader, ncols=80):
#             logits = model.inference(_data["image"].cuda())
#
#             for j in range(logits.shape[0]):
#                 gt = cv2.imread(_data["label_path"][j], cv2.IMREAD_GRAYSCALE)
#                 if gt is None:
#                     raise FileNotFoundError(_data["label_path"][j])
#                 H, W = gt.shape
#
#                 logit_single = logits[j : j + 1]
#                 logit_up = F.interpolate(
#                     logit_single, size=(H, W), mode="bilinear", align_corners=False
#                 )
#
#                 if logit_up.shape[1] == 2:
#                     p = torch.softmax(logit_up, dim=1)[0, 1].cpu().numpy()
#                 else:
#                     p = torch.sigmoid(logit_up[0, 0]).cpu().numpy()
#
#                 pred = np.clip(np.round(p * 255.0), 0, 255).astype(np.uint8)
#
#                 if save_dir is not None:
#                     name = os.path.splitext(_data["fname"][j])[0] + ".png"
#                     Image.fromarray(pred).save(os.path.join(save_dir, name))
#
#                 FM.step(pred=pred, gt=gt)
#                 WFM.step(pred=pred, gt=gt)
#                 SM.step(pred=pred, gt=gt)
#                 EM.step(pred=pred, gt=gt)
#                 M.step(pred=pred, gt=gt)
#
#     results = {
#         "MAE":       '%.4f' % M.get_results()["mae"],
#         "meanEm":    '%.4f' % EM.get_results()["em"]["curve"].mean(),
#         "meanFm":    '%.4f' % FM.get_results()["fm"]["curve"].mean(),
#         "Smeasure":  '%.4f' % SM.get_results()["sm"],
#         "wFmeasure": '%.4f' % WFM.get_results()["wfm"],
#         "adpEm":     '%.4f' % EM.get_results()["em"]["adp"],
#         "adpFm":     '%.4f' % FM.get_results()["fm"]["adp"],
#         "maxEm":     '%.4f' % EM.get_results()["em"]["curve"].max(),
#         "maxFm":     '%.4f' % FM.get_results()["fm"]["curve"].max(),
#     }
#     print("Test Results:", results)
#
#     return results
#
# def train(opt):
#     train_loader = DataLoader(opt)
#     train_data = train_loader.load_data()
#     print("#training images = %d" % len(train_loader))
#
#     opt.phase = "test"
#     test_loader = DataLoader(opt)
#     test_data = test_loader.load_data()
#     print("#testing images = %d" % len(test_loader))
#     opt.phase = "train"
#
#     net = create_model(opt)
#     optimizer = net.optimizer
#     schedular = net.schedular
#
#     assert opt.epoch_val <= opt.num_epochs, \
#         "epoch_val(%d) 不能大于 num_epochs(%d)" % (opt.epoch_val, opt.num_epochs)
#
#     log_path = os.path.join(net.save_dir, "record.txt")
#     if not os.path.exists(log_path):
#         with open(log_path, "a", encoding="utf-8") as f:
#             f.write("# Record of training metrics\n")
#             f.write("# name: %s | backbone: %s\n" % (opt.name, opt.backbone))
#             f.write("# time,epoch,train_loss,lr,metrics(json)\n")
#
#     print("--- start training ---")
#     previous_best = 0.0
#
#     for epoch in range(1, opt.num_epochs + 1):
#         print("\n==> Name %s, Epoch %i, previous best = %.6f"
#               % (opt.name, epoch, previous_best))
#
#         tbar = tqdm(train_data, ncols=80)
#         net.train()
#         _loss = 0.0
#         last_lr = optimizer.param_groups[0]["lr"]
#         i = 0
#         for i, data in enumerate(tbar):
#             _, loss = net(data["image"].cuda(), data["label"].cuda())
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             _loss += loss.item()
#             last_lr = optimizer.param_groups[0]["lr"]
#             tbar.set_description("Loss: %.3f" % (_loss / (i + 1)))
#
#         schedular.step()
#
#         train_stats = {"loss": _loss / max(1, i + 1), "lr": last_lr}
#
#         if epoch >= opt.epoch_val:
#             scores = evaluate(net, test_data, verbose=False)
#             print("(epoch %d) " % epoch
#                   + " ".join("%s: %.6f" % (k, v) for k, v in scores.items()))
#
#             # current_score = scores["wFmeasure"]
#             current_score = scores["Smeasure"]
#             if current_score >= previous_best:
#                 net.save(opt.name, opt.backbone)
#                 previous_best = current_score
#
#             append_log_line(log_path, epoch, train_stats, scores)
#         else:
#             append_log_line(log_path, epoch, train_stats, {"Message": "Skipped eval"})
#
#     print("Done!")
#
#
# if __name__ == "__main__":
#     opt = Options().parse()
#     setup_seed(seed=opt.seed)
#     train(opt)

import os
import json
import random
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
from util.SOD_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


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


def evaluate(model, data_loader, save_dir=None):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for _data in tqdm(data_loader, ncols=80):
            logits = model.inference(_data["image"].cuda())

            for j in range(logits.shape[0]):
                gt = cv2.imread(_data["label_path"][j], cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    raise FileNotFoundError(_data["label_path"][j])
                H, W = gt.shape

                logit_single = logits[j: j + 1]
                logit_up = F.interpolate(
                    logit_single, size=(H, W), mode="bilinear", align_corners=False
                )

                if logit_up.shape[1] == 2:
                    p = torch.softmax(logit_up, dim=1)[0, 1].cpu().numpy()
                else:
                    p = torch.sigmoid(logit_up[0, 0]).cpu().numpy()

                pred = np.clip(np.round(p * 255.0), 0, 255).astype(np.uint8)

                if save_dir is not None:
                    name = os.path.splitext(_data["fname"][j])[0] + ".png"
                    Image.fromarray(pred).save(os.path.join(save_dir, name))

                FM.step(pred=pred, gt=gt)
                WFM.step(pred=pred, gt=gt)
                SM.step(pred=pred, gt=gt)
                EM.step(pred=pred, gt=gt)
                M.step(pred=pred, gt=gt)

    results = {
        "MAE": '%.4f' % M.get_results()["mae"],
        "meanEm": '%.4f' % EM.get_results()["em"]["curve"].mean(),
        "meanFm": '%.4f' % FM.get_results()["fm"]["curve"].mean(),
        "Smeasure": '%.4f' % SM.get_results()["sm"],
        "wFmeasure": '%.4f' % WFM.get_results()["wfm"],
        "adpEm": '%.4f' % EM.get_results()["em"]["adp"],
        "adpFm": '%.4f' % FM.get_results()["fm"]["adp"],
        "maxEm": '%.4f' % EM.get_results()["em"]["curve"].max(),
        "maxFm": '%.4f' % FM.get_results()["fm"]["curve"].max(),
    }
    print("Test Results:", results)

    return results


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

    # 用于维护 S-measure 最好的前三个权重：元素为 (score, file_path)
    best_checkpoints = []

    for epoch in range(1, opt.num_epochs + 1):
        print(f"\n==> Name {opt.name}, Epoch {epoch}, Top-3 Smeasure tracking active")

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
            scores = evaluate(net, test_data)
            print("(epoch %d) " % epoch
                  + " ".join("%s: %s" % (k, v) for k, v in scores.items()))

            current_score = float(scores["Smeasure"])

            # 构造当前 epoch 的权重文件名
            save_filename = f"{opt.name}_{opt.backbone}_epoch{epoch}_sm{current_score:.4f}.pth"
            save_path = os.path.join(net.save_dir, save_filename)

            # 判断是否进入前三名（或者目前记录的权重少于 3 个）
            if len(best_checkpoints) < 3 or current_score > best_checkpoints[0][0]:
                # 保存当前模型权重
                torch.save({
                    "network": net.model.cpu().state_dict(),
                    "optimizer": net.optimizer.state_dict(),
                }, save_path)
                if torch.cuda.is_available():
                    net.model.cuda()

                best_checkpoints.append((current_score, save_path))
                # 按分数从小到大排序
                best_checkpoints.sort(key=lambda x: x[0])

                # 如果超过 3 个，剔除并删除分数最低的那个权重文件
                if len(best_checkpoints) > 3:
                    worst_score, worst_path = best_checkpoints.pop(0)
                    if os.path.exists(worst_path):
                        os.remove(worst_path)
                        print(f"--> Top-3 full, removed previous lower-score ckpt: {os.path.basename(worst_path)}")

                print(f"--> New Top-3 saved! Current top scores: {[f'{s:.4f}' for s, _ in best_checkpoints]}")

            append_log_line(log_path, epoch, train_stats, scores)
        else:
            append_log_line(log_path, epoch, train_stats, {"Message": "Skipped eval"})

    print("Done! Top-3 best weights are safely stored.")


if __name__ == "__main__":
    opt = Options().parse()
    setup_seed(seed=opt.seed)
    train(opt)