import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
from util.SOD_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


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

                logit_single = logits[j : j + 1]
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
        "MAE":       '%.4f' % M.get_results()["mae"],
        "meanEm":    '%.4f' % EM.get_results()["em"]["curve"].mean(),
        "meanFm":    '%.4f' % FM.get_results()["fm"]["curve"].mean(),
        "Smeasure":  '%.4f' % SM.get_results()["sm"],
        "wFmeasure": '%.4f' % WFM.get_results()["wfm"],
        "adpEm":     '%.4f' % EM.get_results()["em"]["adp"],
        "adpFm":     '%.4f' % FM.get_results()["fm"]["adp"],
        "maxEm":     '%.4f' % EM.get_results()["em"]["curve"].max(),
        "maxFm":     '%.4f' % FM.get_results()["fm"]["curve"].max(),
    }
    print("Test Results:", results)

    return results


if __name__ == "__main__":
    opt = Options().parse()
    opt.phase = "test"

    opt.ckpt_name = "SD-7_resnet34_epoch66_sm0.8969.pth"  # 换成你实际的权重文件名

    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    print("#testing images = %d" % len(test_loader))

    opt.load_pretrain = True
    model = create_model(opt)

    save_dir = os.path.join(opt.checkpoint_dir, opt.name, "pred") if opt.save_test else None

    print("=" * 60)
    print("%s Test Metrics:" % opt.name)
    print("=" * 60)
    evaluate(model, test_data, save_dir=save_dir)
    print("=" * 60)