"""
掩码生成工具：
  - Ms（软笔画掩码）：从原图与 GT 的像素差值 + SAF 算法生成
  - Mb（文本块掩码）：优先使用 box_label_txt 精确标注；无标注时退回像素差值+膨胀
"""
import os

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


def generate_mb_from_boxes(txt_path: str,
                            crop_x1: int, crop_y1: int,
                            crop_x2: int, crop_y2: int,
                            patch_size: int) -> np.ndarray:
    """
    从 box_label_txt 标注文件生成文本块掩码 Mb_gt。

    标注格式（每行）：x1,y1,x2,y2,x3,y3,x4,y4, class
        四个角点为顺时针或逆时针排列的四边形，坐标为原始完整图像空间。

    Args:
        txt_path:  对应图像的 box_label_txt 文件路径
        crop_x1/y1/x2/y2: 当前 patch 在原图中的像素范围
        patch_size: 输出掩码边长（正方形）

    Returns:
        Mb_gt: uint8 {0, 1}，shape (patch_size, patch_size)
    """
    Mb = np.zeros((patch_size, patch_size), dtype=np.uint8)

    if not os.path.exists(txt_path):
        return Mb

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 8:
            continue
        try:
            coords = [int(p.strip()) for p in parts[:8]]
        except ValueError:
            continue

        # 四个角点 (x, y)
        pts = np.array(coords, dtype=np.float32).reshape(4, 2)

        # 快速过滤：包围盒与 patch 不相交则跳过
        bx_min, by_min = pts[:, 0].min(), pts[:, 1].min()
        bx_max, by_max = pts[:, 0].max(), pts[:, 1].max()
        if bx_max <= crop_x1 or bx_min >= crop_x2:
            continue
        if by_max <= crop_y1 or by_min >= crop_y2:
            continue

        # 坐标平移到 patch 局部空间，并裁剪到 [0, patch_size-1]
        pts[:, 0] -= crop_x1
        pts[:, 1] -= crop_y1
        pts = np.clip(pts, 0, patch_size - 1).astype(np.int32)

        cv2.fillPoly(Mb, [pts], 1)

    return Mb
