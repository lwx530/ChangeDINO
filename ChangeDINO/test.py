import torch
import os
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from util.WPFormer_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from option import Options
from data.cd_dataset import DataLoader
from model.create_ChangeDINO import create_model
from util.visualize import visualize_dino_adapter


if __name__ == "__main__":
    opt = Options().parse()
    opt.phase = "test"
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    # 2. 创建可视化保存目录（关键：新增，与预测结果目录区分）
    # vis_save_dir = os.path.join(opt.checkpoint_dir, opt.name, "feature_vis")
    # os.makedirs(vis_save_dir, exist_ok=True)

    tbar = tqdm(test_data, ncols=80)
    # total_iters = test_size
    # running_metric = ConfuseMatrixMeter(n_class=2)
    # running_metric.clear()

    # 初始化WPFormer指标
    M = MAE()
    EM = Emeasure()
    FM = Fmeasure()
    SM = Smeasure()
    WFM = WeightedFmeasure()

    test_save_path = os.path.join(opt.checkpoint_dir, opt.name, "pred")
    if opt.save_test and not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            # img_tensor = _data["image"].cuda()  # 原图张量 [B,3,H,W]
            val_pred = model.inference(_data["image"].cuda())
            # val_pred, dino_feats, adapter_feats = model.inference(_data["image"].cuda())
            # update metric
            val_target = _data["label"].detach()

            '''# 1. 获取概率图
            if val_pred.shape[1] == 2:
                val_pred_prob = torch.softmax(val_pred.detach(), dim=1)[:, 1]
                
            else:
                val_pred_prob = torch.sigmoid(val_pred.detach().squeeze(1))'''

            # ===== 重新安全计算概率 =====
            scale = 1.8

            logits = val_pred.detach() * scale

            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)
                val_pred_prob = probs[:, 1, :, :]  # 明确取第2类
            else:
                val_pred_prob = torch.sigmoid(logits[:, 0, :, :])

            # print("val_pred_prob shape:", val_pred_prob.shape)

            # 2. 确保标签是二维
            if val_target.dim() == 4:
                val_target = val_target.squeeze(1)

            # 3. 更新WPFormer指标
            for j in range(val_pred_prob.shape[0]):
                pred_np = val_pred_prob[j].cpu().numpy()
                target_np = val_target[j].cpu().numpy()

                pred_uint8 = (pred_np * 255).astype(np.uint8)
                gt_uint8 = (target_np * 255).astype(np.uint8)

                M.step(pred_uint8, gt_uint8, normalize=True)
                EM.step(pred_uint8, gt_uint8, normalize=True)
                FM.step(pred_uint8, gt_uint8, normalize=True)
                SM.step(pred_uint8, gt_uint8, normalize=True)
                WFM.step(pred_uint8, gt_uint8, normalize=True)

            if opt.save_test:
                # 用概率图生成二值图
                val_pred_binary = (val_pred_prob > 0.5).long()
                for j in range(val_pred_binary.shape[0]):
                    pred = Image.fromarray((val_pred_binary[j].cpu().detach().numpy() * 255).astype("uint8"))
                    pred.save(
                        os.path.join(test_save_path, _data["fname"][j])
                    )

            '''# 4. 特征可视化核心逻辑（关键：新增，批量处理每个batch的每张图片）
            for j in range(img_tensor.shape[0]):
                # 提取单张图片的所有数据（去除batch维度）
                img_single = img_tensor[j]  # 原图 [3,H,W]
                dino_feats_single = [f[j] for f in dino_feats]  # DINOv3 4层特征，每个[1024,H_i,W_i]
                adapter_feats_single = [f[j] for f in adapter_feats]  # Adapter 4层特征，每个[256,H_i,W_i]
                img_name = os.path.splitext(_data["fname"][j])[0]  # 去除后缀，作为可视化文件名

                # 调用可视化函数
                visualize_dino_adapter(
                    img_ori=img_single,
                    dino_feats=dino_feats_single,
                    adapter_feats=adapter_feats_single,
                    save_dir=vis_save_dir,
                    img_name=img_name,
                    dino_sizes=[64, 32, 16, 8]  # 与Encoder中DefectAdapter的sizes完全一致
                )'''

        # 获取WPFormer指标结果
        M_result = M.get_results()
        EM_result = EM.get_results()
        FM_result = FM.get_results()
        SM_result = SM.get_results()
        WFM_result = WFM.get_results()

        # ====== 计算 Precision / Recall ======
        import numpy as np

        all_precisions = np.array(FM.precisions)  # [N, 256]
        all_recalls = np.array(FM.recalls)  # [N, 256]
        all_fms = np.array(FM.changeable_fms)  # [N, 256]

        mean_precision_curve = all_precisions.mean(axis=0)
        mean_recall_curve = all_recalls.mean(axis=0)
        mean_f_curve = all_fms.mean(axis=0)

        mean_precision = mean_precision_curve.mean()
        mean_recall = mean_recall_curve.mean()

        best_idx = np.argmax(mean_f_curve)

        best_precision = mean_precision_curve[best_idx]
        best_recall = mean_recall_curve[best_idx]

        print("\n" + "=" * 60)
        print("Precision / Recall Analysis:")
        print("=" * 60)
        print(f"Mean Precision: {mean_precision:.6f}")
        print(f"Mean Recall: {mean_recall:.6f}")
        print(f"Best Threshold Precision: {best_precision:.6f}")
        print(f"Best Threshold Recall: {best_recall:.6f}")
        print("=" * 60)

        val_scores = {
            'MAE': M_result['mae'],
            'Emeasure': EM_result['em']['adp'],
            'Fmeasure': FM_result['fm']['adp'],
            'Smeasure': SM_result['sm'],
            'wFmeasure': WFM_result['wfm']
        }

        # 输出结果
        print("\n" + "=" * 60)
        print("WPFormer Test Metrics:")
        print("=" * 60)
        for k, v in val_scores.items():
            print(f"{k}: {v:.6f}")
        print("=" * 60)
