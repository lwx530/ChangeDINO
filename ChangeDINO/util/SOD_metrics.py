# -*- coding: utf-8 -*-
import warnings

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist


import numpy as np

# the different implementation of epsilon (extreme min value) between numpy and matlab
EPS = np.spacing(1)
TYPE = np.float64


def validate_and_normalize_input(pred: np.ndarray, gt: np.ndarray, normalize: bool = True):     # 执行输入验证和归一化
    """Performs input validation and normalization."""
    # Validate input shapes
    if pred.shape != gt.shape:      # 确保预测结果和真实标签具有相同的维度形状
        raise ValueError(           # 如果不一致，抛出详细的错误信息
            f"Shape mismatch between prediction ({pred.shape}) and ground truth ({gt.shape})"
        )

    # Handle normalization
    if normalize:       # 启用归一化时的处理
        pred, gt = prepare_data(pred, gt)       # 调用prepare_data函数进行完整的数据预处理
    else:       # 非归一化时的验证
        # Validate prediction data type and range
        if pred.dtype not in (np.float32, np.float64):          # 要求预测数组必须是32位或64位浮点数
            raise TypeError(f"Prediction array must be float32 or float64, got {pred.dtype}")       # 如果是不兼容的类型（如uint8, int16等），抛出类型错误
        if not (0 <= pred.min() and pred.max() <= 1):       # 检查预测值是否在[0, 1]范围内
            raise ValueError("Prediction values must be in range [0, 1]")
        # Validate ground truth type
        if gt.dtype != bool:        # 要求真实标签必须是布尔类型(bool)
            raise TypeError(f"Ground truth must be boolean, got {gt.dtype}")

    return pred, gt     # 返回处理后的预测值和标签


def prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """A numpy-based function for preparing `pred` and `gt`.

    - for `pred`, it looks like `mapminmax(im2double(...))` of matlab;
    - `gt` will be binarized by 128.

    Args:
        pred (np.uint8): Prediction, gray scale image.
        gt (np.uint8): Ground truth, gray scale image.

    Returns:
        tuple: pred (np.float64), gt (bool)
    """
    gt = gt > 128       # 标签二值化，> 128: 像素值大于128的设为True(前景)，否则False(背景)
    # im2double, mapminmax
    pred = pred / 255       # 将uint8的[0,255]范围缩放到[0,1]浮点数范围
    if pred.max() != pred.min():        # 条件判断：避免除零错误（当图像为常数值时）
        pred = (pred - pred.min()) / (pred.max() - pred.min())      # 将数据线性变换到[0,1]范围，公式：(x - min) / (max - min)
    return pred, gt     # 返回处理后的预测值(float64)和二值标签(bool)


def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:      # 计算自适应阈值
    """Return an adaptive threshold, which is equal to twice the mean of `matrix`.

    Args:
        matrix (np.ndarray): a data array
        max_value (float, optional): the upper limit of the threshold. Defaults to 1.

    Returns:
        float: `min(2 * matrix.mean(), max_value)`
    """
    return min(2 * matrix.mean(), max_value)        # min(..., max_value): 确保阈值不超过上限
class Fmeasure(object):
    def __init__(self, beta: float = 0.3):
        """F-measure for SOD.

        ```
        @inproceedings{Fmeasure,
            title={Frequency-tuned salient region detection},
            author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and S{\"u}sstrunk, Sabine},
            booktitle=CVPR,
            number={CONF},
            pages={1597--1604},
            year={2009}
        }
        ```

        Args:
            beta (float): the weight of the precision
        """
        warnings.warn("This class will be removed in the future, please use FmeasureV2 instead!")

        self.beta = beta        # beta控制F-measure中精确率和召回率的平衡，当beta < 1时，更重视召回率(Recall)
        self.precisions = []    # 存储每个样本在不同阈值下的精确率曲线
        self.recalls = []       # 存储每个样本在不同阈值下的召回率曲线
        self.adaptive_fms = []  # 存储每个样本的自适应F值
        self.changeable_fms = []    # 存储每个样本在不同阈值下的F值曲线

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)        # 调用validate_and_normalize_input函数数据验证和归一化

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)        # 使用自适应阈值计算F值
        self.adaptive_fms.append(adaptive_fm)       # 将单个F值添加到adaptive_fms列表

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)     # 计算256个阈值下的精确率、召回率和F值
        self.precisions.append(precisions)      # 将三个曲线数组分别添加到对应的列表
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:       # 计算自适应F-measure值
        """Calculate the adaptive F-measure.

        Returns:
            float: adaptive_fm
        """
        # ``np.count_nonzero`` is faster and better
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)      # 获取自适应阈值
        binary_predcition = pred >= adaptive_threshold      # 生成二值预测图，将预测图中≥阈值的像素设为True，否则False
        area_intersection = binary_predcition[gt].sum()     # 计算真正例数量，使用gt作为掩码索引binary_prediction
        if area_intersection == 0:      # 如果真正例数量为0，直接返回F值=0
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_predcition)   # 计算精确率，分母：预测为正例的总数（binary_prediction中True的数量）
            rec = area_intersection / np.count_nonzero(gt)      # 分母：真实正例的总数（gt中True的数量）
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)     # 计算F-measure
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:        # 计算256个阈值下的精确率、召回率和F值曲线
        """Calculate the corresponding precision and recall when the threshold changes from 0 to 255.

        These precisions and recalls can be used to obtain the mean F-measure, maximum F-measure,
        precision-recall curve and F-measure-threshold curve.

        For convenience, `changeable_fms` is provided here, which can be used directly to obtain
        the mean F-measure, maximum F-measure and F-measure-threshold curve.

        Returns:
            tuple: (precisions, recalls, changeable_fms)
        """
        # 1. 获取预测结果在真值前背景区域中的直方图
        pred = (pred * 255).astype(np.uint8)  # 预测图转换，将[0,1]浮点数转换为[0,255]整数，便于使用256个阈值进行二值化
        bins = np.linspace(0, 256, 257)     # 创建直方图bins，创建257个边界点：[0, 1, 2, ..., 255, 256]，对应256个区间
        fg_hist, _ = np.histogram(pred[gt], bins=bins)  # 最后一个bin为[255, 256]，真值前景区域的预测值分布
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)   # 真值背景区域的预测值分布
        # 2. 使用累积直方图（Cumulative Histogram）获得对应真值前背景中大于不同阈值的像素数量
        # 这里使用累加（cumsum）就是为了一次性得出 >=不同阈值 的像素数量, 这里仅计算了前景区域
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        # 3. 使用不同阈值的结果计算对应的precision和recall
        # p和r的计算的真值是pred==1&gt==1，二者仅有分母不同，分母前者是pred==1，后者是gt==1
        # 为了同时计算不同阈值的结果，这里使用hsitogram&flip&cumsum 获得了不同各自的前景像素数量
        TPs = fg_w_thrs     # 不同阈值下的真正例数量
        Ps = fg_w_thrs + bg_w_thrs      # 预测正例总数
        # 为防止除0，这里针对除0的情况分析后直接对于0分母设为1，因为此时分子必为0
        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)        # 标签中前景像素总数
        # TODO: T=0 或者 特定阈值下fg_w_thrs=0或者bg_w_thrs=0，这些都会包含在TPs[i]=0的情况中，
        #  但是这里使用TPs不便于处理列表
        precisions = TPs / Ps
        recalls = TPs / T

        numerator = (1 + self.beta) * precisions * recalls      # 分子
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)     # 分母防除零处理
        changeable_fms = numerator / denominator        # F值计算
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:      # F-measure类的结果汇总函数
        """Return the results about F-measure.

        Returns:
            dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))
        """
        adaptive_fm = np.mean(np.array(self.adaptive_fms, TYPE))        # 计算自适应F值的平均值
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=TYPE), axis=0)      # 计算F值曲线的平均值
        precision = np.mean(np.array(self.precisions, dtype=TYPE), axis=0)  # N, 256,计算精确率曲线的平均值
        recall = np.mean(np.array(self.recalls, dtype=TYPE), axis=0)  # N, 256,计算256个阈值下的平均召回率
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))


class MAE(object):
    def __init__(self):
        """MAE(mean absolute error) for SOD.

        ```
        @inproceedings{MAE,
            title={Saliency filters: Contrast based filtering for salient region detection},
            author={Perazzi, Federico and Kr{\"a}henb{\"u}hl, Philipp and Pritch, Yael and Hornung, Alexander},
            booktitle=CVPR,
            pages={733--740},
            year={2012}
        }
        ```
        """
        self.maes = []      # 用于累积所有样本的MAE值

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)        # 验证和归一化输入数据

        mae = self.cal_mae(pred, gt)        # 计算MAE: 调用cal_mae方法
        self.maes.append(mae)       # 将单个样本的MAE添加到列表

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:      # 核心计算
        """Calculate the mean absolute error.

        Returns:
            np.ndarray: mae
        """
        mae = np.mean(np.abs(pred - gt))        # 计算每个像素的误差,取绝对值, 计算所有像素的平均值
        return mae

    def get_results(self) -> dict:
        """Return the results about MAE.

        Returns:
            dict(mae=mae)
        """
        mae = np.mean(np.array(self.maes, TYPE))    # 将所有样本的MAE转换为numpy数组,计算平均值作为最终评估结果
        return dict(mae=mae)


class Smeasure(object):         # 计算显著性检测的结构相似性度量
    def __init__(self, alpha: float = 0.5):
        """S-measure(Structure-measure) of SOD.

        ```
        @inproceedings{Smeasure,
            title={Structure-measure: A new way to eval foreground maps},
            author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
            booktitle=ICCV,
            pages={4548--4557},
            year={2017}
        }
        ```

        Args:
            alpha: the weight for balancing the object score and the region score
        """
        self.sms = []       # 空列表，用于存储所有样本的S-measure值
        self.alpha = alpha      # 存储平衡权重参数

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):       # 处理单个样本
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)     # 数据验证和归一化,检查pred和gt的形状是否一致
        # 将pred从[0,255]归一化到[0,1]范围;将gt二值化（>128为True，否则False）
        sm = self.cal_sm(pred, gt)      # 调用cal_sm方法计算当前样本的S-measure值
        self.sms.append(sm)     # 将当前样本的S-measure值添加到结果列表中

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:        # 主计算函数
        """Calculate the S-measure.

        Returns:
            s-measure
        """
        y = np.mean(gt)     # 计算标签的均值
        if y == 0:          # 全背景图像 (y == 0)
            sm = 1 - np.mean(pred)      # 如果预测接近0，得分接近1（完美）；如果预测接近1，得分接近0（最差）
        elif y == 1:        # 全前景图像 (y == 1)
            sm = np.mean(pred)      # 如果预测接近1，得分接近1（完美）；如果预测接近0，得分接近0（最差）
        else:       # 正常图像 (包含前景和背景)
            object_score = self.object(pred, gt) * self.alpha       # 对象级结构相似性，使用self.alpha进行加权平衡
            region_score = self.region(pred, gt) * (1 - self.alpha)     # 区域级结构相似性
            sm = max(0, object_score + region_score)     # max(0, ...)确保结果非负
        return sm

    def s_object(self, x: np.ndarray) -> float:    # 计算单个区域的对象相似性
        mean = np.mean(x)       # 计算区域均值,反映区域的整体亮度水平
        std = np.std(x, ddof=1)     # 计算区域标准差,ddof=1表示使用无偏估计（除以N-1而不是N）,反映区域内部的变化程度
        score = 2 * mean / (np.power(mean, 2) + 1 + std + EPS)      # 对象相似性公式,score = 2 × mean / (mean² + 1 + std + EPS)
        return score

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:        # 计算对象级结构相似性得分
        """Calculate the object score."""
        gt_mean = np.mean(gt)       # 前景像素占总像素的比例
        fg_score = self.s_object(pred[gt]) * gt_mean      # 计算前景得分
        # pred[gt]提取前景区域,调用s_object方法计算前景区域的结构相似性,用前景比例对前景得分进行加权
        bg_score = self.s_object((1 - pred)[~gt]) * (1 - gt_mean)     # 计算背景得分
        # 计算背景预测值,[~gt]提取背景区域,计算背景对象相似性,加权
        object_score = fg_score + bg_score      # 将加权后的前景得分和背景得分相加
        return object_score     # 返回对象得分

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:        # 区域得分计算
        """Calculate the region score."""
        h, w = gt.shape     # 获取图像的高度和宽度
        area = h * w        # 获取图像的总像素数

        # 计算前景质心坐标
        if np.count_nonzero(gt) == 0:       # 无前景像素
            cy, cx = np.round(h / 2), np.round(w / 2)       # 如果图像全背景，使用图像中心作为质心
        else:               # 有前景像素
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            cy, cx = np.argwhere(gt).mean(axis=0).round()       # 返回所有前景像素的坐标数组，计算坐标的平均值，得到质心
        # To ensure consistency with the matlab code, one is added to the centroid coordinate,
        # so there is no need to use the redundant addition operation when dividing the region later,
        # because the sequence generated by ``1:X`` in matlab will contain ``X``.
        cy, cx = int(cy) + 1, int(cx) + 1       # 调整质心坐标，+1是为了与Matlab代码保持一致

        # 计算四个区域的权重，每个区域的权重等于其面积占总面积的比例，确保四个权重之和为1
        w_lt = cx * cy / area
        w_rt = cy * (w - cx) / area
        w_lb = (h - cy) * cx / area
        w_rb = 1 - w_lt - w_rt - w_lb
        # 计算四个区域的得分，分别计算每个区域的SSIM相似性，用区域权重进行加权
        score_lt = self.ssim(pred[0:cy, 0:cx], gt[0:cy, 0:cx]) * w_lt
        score_rt = self.ssim(pred[0:cy, cx:w], gt[0:cy, cx:w]) * w_rt
        score_lb = self.ssim(pred[cy:h, 0:cx], gt[cy:h, 0:cx]) * w_lb
        score_rb = self.ssim(pred[cy:h, cx:w], gt[cy:h, cx:w]) * w_rb
        return score_lt + score_rt + score_lb + score_rb        # 返回加权后的区域总得分

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:      # 结构相似性计算
        """Calculate the ssim score."""
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)       # 预测区域的均值
        y = np.mean(gt)         # 真实标签区域的均值

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)     # 预测区域的方差（无偏估计）
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)       # 真实标签区域的方差
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)      #  预测和真实标签的协方差
        # 计算SSIM的分子和分母
        alpha = 4 * x * y * sigma_xy        # 结合了亮度、对比度和结构信息
        beta = (x**2 + y**2) * (sigma_x + sigma_y)      # 标准化项
        # 计算最终得分
        if alpha != 0:
            score = alpha / (beta + EPS)        # 正常计算
        elif alpha == 0 and beta == 0:
            score = 1       # 两个区域都是常数，完全相似
        else:
            score = 0       # 其他情况
        return score

    def get_results(self) -> dict:
        """Return the results about S-measure.

        Returns:
            dict(sm=sm)
        """
        sm = np.mean(np.array(self.sms, dtype=TYPE))        # 将所有样本的S-measure值转换为数组，计算平均值作为最终结果
        return dict(sm=sm)      # 返回字典格式的结果


class Emeasure(object):
    def __init__(self):
        """E-measure(Enhanced-alignment Measure) for SOD.

        More details about the implementation can be found in https://www.yuque.com/lart/blog/lwgt38

        ```
        @inproceedings{Emeasure,
            title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
            author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
            booktitle=IJCAI,
            pages="698--704",
            year={2018}
        }
        ```
        """
        self.adaptive_ems = []          # 存储每个样本的自适应E值
        self.changeable_ems = []        # 存储每个样本的可变阈值E值曲线

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):   # 对一对预测图和真实标签图进行E-measure统计
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)    # 数据验证和归一化，检查pred和gt的形状一致性

        self.gt_fg_numel = np.count_nonzero(gt)     # 计算真实前景像素数量，gt是二值数组(True/False)，np.count_nonzero统计True的数量
        self.gt_size = gt.shape[0] * gt.shape[1]        # 计算图像总像素数，高度 × 宽度

        changeable_ems = self.cal_changeable_em(pred, gt)   # 计算256个阈值下的E值曲线
        self.changeable_ems.append(changeable_ems)      # 添加到changeable_ems列表
        adaptive_em = self.cal_adaptive_em(pred, gt)        # 使用自适应阈值计算单个E值
        self.adaptive_ems.append(adaptive_em)       # 添加到adaptive_ems列表

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:       #  自适应E值计算
        """Calculate the adaptive E-measure.

        Returns:
            adaptive_em
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)      # 获取自适应阈值
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)        #  使用自适应阈值计算E值
        return adaptive_em      # 返回自适应E值

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:        # 可变阈值E值计算
        """Calculate the changeable E-measure, which can be used to obtain the mean E-measure, the maximum E-measure and the E-measure-threshold curve.

        Returns:
            changeable_ems
        """
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)     # 使用累积直方图方法计算E值曲线
        return changeable_ems       # 返回E值曲线数组

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:   # 计算特定阈值下的E-measure值
        """Calculate the E-measure corresponding to the specific threshold.

        Variable naming rules within the function:
        `[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]`

        If only `pred` or `gt` is considered, another corresponding attribute location is replaced with '`_`'.
        """
        binarized_pred = pred >= threshold      #  二值化预测图，True表示预测前景，False表示预测背景
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)     # 计算真正例(TP)，预测前景且真实前景的像素，统计True的数量
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)    # 计算假正例(FP)，预测前景但真实背景的像素

        fg___numel = fg_fg_numel + fg_bg_numel      # 计算预测前景总数
        bg___numel = self.gt_size - fg___numel      # 计算预测背景总数

        if self.gt_fg_numel == 0:       # 全背景图像
            enhanced_matrix_sum = bg___numel        # 使用预测背景数作为增强矩阵和
        elif self.gt_fg_numel == self.gt_size:      # 全前景图像，真实标签中所有像素都是前景
            enhanced_matrix_sum = fg___numel        # 使用预测前景数作为增强矩阵和
        else:       # 正常图像（包含前景和背景）
            parts_numel, combinations = self.generate_parts_numel_combinations(     # 生成四个区域的像素数和组合
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )       # parts_numel: 四个区域的像素数,combinations: 四个区域的去均值组合

            results_parts = []      # 遍历四个区域计算增强值
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):      # 四个区域：真正例、假正例、假负例、真负例
                align_matrix_value = (      # 计算对齐矩阵值
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4       # 计算增强矩阵值
                results_parts.append(enhanced_matrix_value * part_numel)        # 加权累加,将增强值乘以该区域的像素数
            enhanced_matrix_sum = sum(results_parts)        # 计算增强矩阵总和,四个区域的加权增强值之和

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)       # 归一化得到E-measure
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:  # 计算0-255所有阈值下的E-measure曲线
        """Calculate the E-measure corresponding to the threshold that varies from 0 to 255..

        Variable naming rules within the function:
        `[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]`

        If only `pred` or `gt` is considered, another corresponding attribute location is replaced with '`_`'.
        """
        pred = (pred * 255).astype(np.uint8)        # 转换预测值为整数
        bins = np.linspace(0, 256, 257)     # 创建直方图区间
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)   # 计算真正例区域直方图
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)      # 计算假正例区域直方图
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)     # 计算真正例累积直方图
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)     # 计算假正例累积直方图

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs     # 计算预测前景总数
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs        # 计算预测背景总数

        if self.gt_fg_numel == 0:       # 全背景图像
            enhanced_matrix_sum = bg___numel_w_thrs     # 使用预测背景数作为增强矩阵和
        elif self.gt_fg_numel == self.gt_size:      # 全前景图像
            enhanced_matrix_sum = fg___numel_w_thrs     # 直接使用预测前景数作为增强矩阵和
        else:           # 正常图像
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(      # 生成四个区域的像素数和组合
                fg_fg_numel=fg_fg_numel_w_thrs,         # 所有参数现在都是长度为256的数组
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)      # 初始化结果数组，用于存储每个区域在每个阈值下的加权增强值
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):       # 遍历四个区域计算
                align_matrix_value = (      # 计算对齐矩阵值（向量化）
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4       # 计算增强矩阵值（向量化）
                results_parts[i] = enhanced_matrix_value * part_numel       # 存储加权结果，增强值 × 像素数
            enhanced_matrix_sum = results_parts.sum(axis=0)     # 计算增强矩阵总和

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)     # 最终E值计算
        return em

    def generate_parts_numel_combinations(      # 生成四个区域的像素数和去均值组合
        self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel
    ):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel        # 计算假负例(FN)数量，真实前景总数 - 真正例 = 假负例
        bg_bg_numel = pred_bg_numel - bg_fg_numel       # 计算真负例(TN)数量

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]      # 组织四个区域的像素数

        mean_pred_value = pred_fg_numel / self.gt_size      # 计算预测均值
        mean_gt_value = self.gt_fg_numel / self.gt_size     # 计算真实均值

        demeaned_pred_fg_value = 1 - mean_pred_value        # 预测前景去均值
        demeaned_pred_bg_value = 0 - mean_pred_value        # 预测背景去均值
        demeaned_gt_fg_value = 1 - mean_gt_value        # 真实前景去均值
        demeaned_gt_bg_value = 0 - mean_gt_value        # 真实背景去均值

        combinations = [            # 组织四个区域的组合
            (demeaned_pred_fg_value, demeaned_gt_fg_value),     # (预测前景, 真实前景) - 真正例区域
            (demeaned_pred_fg_value, demeaned_gt_bg_value),     # (预测前景, 真实背景) - 假正例区域
            (demeaned_pred_bg_value, demeaned_gt_fg_value),     # (预测背景, 真实前景) - 假负例区域
            (demeaned_pred_bg_value, demeaned_gt_bg_value),     # (预测背景, 真实背景) - 真负例区域
        ]
        return parts_numel, combinations        # 返回四个区域的像素数和对应的去均值组合

    def get_results(self) -> dict:      #  汇总所有样本的E-measure结果
        """Return the results about E-measure.

        Returns:
            dict(em=dict(adp=adaptive_em, curve=changeable_em))
        """
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=TYPE))      # 计算平均自适应E值
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=TYPE), axis=0)      # 计算平均E值曲线
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))      # 返回结果字典


class WeightedFmeasure(object):     # 计算显著性检测的加权F值
    def __init__(self, beta: float = 1):
        """Weighted F-measure for SOD.

        ```
        @inproceedings{wFmeasure,
            title={How to eval foreground maps?},
            author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
            booktitle=CVPR,
            pages={248--255},
            year={2014}
        }
        ```

        Args:
            beta (float): the weight of the precision
        """
        self.beta = beta        # 精确率的权重，默认1表示平衡的F1-score
        self.weighted_fms = []      # weighted_fms列表用于累积所有样本的加权F值

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):   # 对一对预测图和真实标签图进行加权F值统计
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)        #  数据验证和归一化

        if np.all(~gt):     # 全背景图像
            wfm = 0     # 直接设置加权F值为0
        else:       # 正常情况计算
            wfm = self.cal_wfm(pred, gt)        # 调用cal_wfm方法计算当前样本的加权F值
        self.weighted_fms.append(wfm)       # 将当前样本的加权F值添加到结果列表中

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:       # 计算加权F-measure值
        """Calculate the weighted F-measure."""
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)        # 计算到背景边界的距离

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)       # 每个像素的预测值与真实值的绝对差异
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)         # 误差传播
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]        #

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)      # 高斯滤波
        EA = convolve(Et, weights=K, mode="constant", cval=0)       # 对传播后的误差进行平滑处理
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)       # 取最小误差，对于前景像素，如果平滑误差小于原始误差，使用平滑误差，否则使用原始误差

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))      # 计算重要性权重
        Ew = MIN_E_EA * B       # 计算加权误差，最小误差 × 重要性权重

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])      # 加权真正例,真实前景像素总数（理想TP）- 前景区域的加权误差和
        FPw = np.sum(Ew[gt == 0])       # 加权假正例

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        # 注意这里使用mask索引矩阵的时候不可使用Ew[gt]，这实际上仅在索引Ew的0维度
        R = 1 - np.mean(Ew[gt == 1])        # 计算加权召回率(R)，1 - 前景平均加权误差
        P = TPw / (TPw + FPw + EPS)         # 加权精确率(P)，TPw / (TPw + FPw)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + EPS)     # 计算加权F值

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:        # 生成与Matlab一致的2D高斯核
        """2D gaussian mask - should give the same result as MATLAB's:
        `fspecial('gaussian',[shape],[sigma])`
        """
        m, n = [(ss - 1) / 2 for ss in shape]       # 计算中心坐标，对于7×7的核：m = (7-1)/2 = 3, n = 3
        y, x = np.ogrid[-m : m + 1, -n : n + 1]     # 创建坐标网格，np.ogrid: 创建开放网格，节省内存
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))      # 计算高斯函数值，高斯公式: exp(-(x² + y²) / (2σ²))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0      # 去除极小值，将小于eps * max的值设为0，提高数值稳定性
        sumh = h.sum()          # 计算总和用于归一化
        if sumh != 0:           # 归一化高斯核
            h /= sumh
        return h        # 返回归一化的高斯核

    def get_results(self) -> dict:      # 汇总所有样本的加权F值结果
        """Return the results about weighted F-measure.

        Returns:
            dict(wfm=weighted_fm)
        """
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=TYPE))      # 计算平均加权F值
        return dict(wfm=weighted_fm)            # 返回结果字典