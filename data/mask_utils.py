"""
掩码生成工具：根据原图与擦除 GT 图的像素差异，生成软笔画掩码 Ms 和文本块掩码 Mb。
"""
import cv2
import numpy as np


def generate_mask_from_pair(Iin: np.ndarray, Igt: np.ndarray,
                             threshold: int = 20,
                             debug: bool = False):
    """
    单块（512×512）软笔画掩码生成。

    Args:
        Iin: 原始图像，(H, W, 3) RGB uint8
        Igt: 擦除 GT 图，(H, W, 3) RGB uint8
        threshold: 判定笔画区域的像素差异阈值，越小掩码越密
        debug: True 时额外返回中间结果字典

    Returns:
        Ms_gt: 软笔画掩码，float32 [0, 1]
        Mb_gt: 文本块掩码，float32 {0, 1}
        debug_info (仅 debug=True 时返回)
    """
    H, W = Iin.shape[:2]

    # 1. 粗笔画掩码：RGB 三通道平均差异 > threshold
    diff = np.abs(Iin.astype(np.int16) - Igt.astype(np.int16)).mean(axis=-1)
    coarse_mask = (diff > threshold).astype(np.uint8)

    # 2. 形态学去噪：开运算去散点 + 闭运算连笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    denoised = cv2.morphologyEx(coarse_mask, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # 3. Mb_gt：对去噪掩码膨胀，覆盖文本块区域
    Mb_gt = cv2.dilate(denoised, kernel, iterations=2).astype(np.float32)

    # 4. Ms_gt：SAF（Stroke Attention Function）软掩码
    skeleton = cv2.erode(denoised, kernel, iterations=1)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    outer_region = cv2.dilate(denoised, kernel_large, iterations=5)

    dist_map = cv2.distanceTransform(outer_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    alpha, L = 3.0, 5.0
    exp_neg_alpha = np.exp(-alpha)
    C = (1 + exp_neg_alpha) / (1 - exp_neg_alpha + 1e-8)

    Ms_gt = np.zeros((H, W), dtype=np.float32)
    mask_skeleton = skeleton > 0
    mask_outer = outer_region > 0
    mask_middle = mask_outer & (~mask_skeleton)

    Ms_gt[mask_skeleton] = 1.0
    if np.any(mask_middle):
        D = np.clip(dist_map[mask_middle], 0, L)
        saf = C * (2.0 / (1.0 + np.exp(-alpha * D / L) + 1e-8) - 1.0)
        Ms_gt[mask_middle] = np.clip(saf, 0.0, 1.0)

    if debug:
        debug_info = {
            'coarse_mask': coarse_mask,
            'denoised_mask': denoised,
            'skeleton': skeleton,
            'outer_region': outer_region,
            'dist_map': dist_map,
            'mask_middle': mask_middle.astype(np.uint8) * 255,
        }
        return Ms_gt, Mb_gt, debug_info

    return Ms_gt, Mb_gt
