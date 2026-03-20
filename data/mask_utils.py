"""
掩码生成工具：
  - Ms（软笔画掩码）：从原图与 GT 的像素差值 + SAF 算法生成
  - Mb（文本块掩码）：优先使用 box_label_txt 精确标注；无标注时退回像素差值+膨胀
"""
import os

import cv2
import numpy as np


def generate_mask_from_pair(Iin, Igt, threshold=20, debug=False):
    """
    单块（512x512）软笔画掩码生成 - 修正版本
    :param Iin: 单块图像 (H,W,3) RGB 0-255
    :param Igt: 单块GT图 (H,W,3) RGB 0-255
    :param threshold: 差异阈值，默认20
    :param debug: 是否返回中间结果用于调试
    :return: Ms_gt(软掩码), Mb_gt(基础掩码), [debug_dict]
    """
    H, W = Iin.shape[:2]

    # ========== 1. 生成粗笔画掩码 ==========
    # 使用int16避免溢出，计算RGB三通道的平均差异
    diff = np.abs(Iin.astype(np.int16) - Igt.astype(np.int16)).mean(axis=-1)
    print(f"Diff统计: min={diff.min()}, max={diff.max()}, 90%={np.percentile(diff, 90)}")
    coarse_mask = (diff > threshold).astype(np.uint8)

    # ========== 2. 形态学去噪 ==========
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 开运算去除散点噪声 + 闭运算连接断裂笔画
    denoised_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_OPEN, kernel)
    denoised_mask = cv2.morphologyEx(denoised_mask, cv2.MORPH_CLOSE, kernel)

    # ========== 3. 生成 Mb_gt（文本块掩码） ==========
    # 论文 Eq.(3): Icomp = Ire·Mb + Iin·(1-Mb)
    # Mb 是文字区域的矩形边框填充（bounding-box region），用于合成时区分文字/背景
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(denoised_mask, connectivity=8)
    Mb_gt = np.zeros((H, W), dtype=np.float32)
    for idx in range(1, num_labels):
        x0 = stats[idx, cv2.CC_STAT_LEFT]
        y0 = stats[idx, cv2.CC_STAT_TOP]
        bw = stats[idx, cv2.CC_STAT_WIDTH]
        bh = stats[idx, cv2.CC_STAT_HEIGHT]
        Mb_gt[y0:y0 + bh, x0:x0 + bw] = 1.0

    # ========== 4. 生成软笔画掩码 Ms_gt ==========
    # 4.1 骨架：笔画收缩1像素（论文要求）
    skeleton = cv2.erode(denoised_mask, kernel, iterations=1)

    # 4.2 外边界区域：原始笔画扩张5像素（3×3 kernel 半径=1px，5次=恰好5px）
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    outer_region = cv2.dilate(denoised_mask, kernel_outer, iterations=5)

    # 4.3 距离变换：计算每个像素到最近笔画像素的距离
    # 在反转掩码上做DT：笔画内部=0，笔画外部=到最近笔画边缘的欧氏距离
    # 这样D(i,j) = L - dist_to_stroke，即到外边界(5px处)的距离，逐笔画独立计算
    # 避免稠密文字时多笔画合并成大区域导致内部距离失真（出现矩形块）
    dist_to_stroke = cv2.distanceTransform(
        (1 - denoised_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 4.4 SAF 参数 (论文: α=3, L=5)
    alpha = 3.0
    L = 5.0
    exp_neg_alpha = np.exp(-alpha)
    C = (1 + exp_neg_alpha) / (1 - exp_neg_alpha + 1e-8)

    # 4.5 向量化计算软掩码
    Ms_gt = np.zeros((H, W), dtype=np.float32)
    mask_skeleton = skeleton > 0
    mask_outer = outer_region > 0

    # D(i,j) = L - dist_to_stroke：笔画内dist=0→D=L=5(SAF≈1)，外边界dist=5→D=0(SAF=0)
    D_map = np.clip(L - dist_to_stroke, 0.0, L)
    exp_term = np.exp(-alpha * D_map[mask_outer] / L)
    saf_vals = C * (2.0 / (1.0 + exp_term + 1e-8) - 1.0)
    Ms_gt[mask_outer] = np.clip(saf_vals, 0.0, 1.0)

    # 骨架强制=1（覆盖SAF结果，确保笔画中心最亮）
    Ms_gt[mask_skeleton] = 1.0

    # ========== 5. 调试信息（可选） ==========
    if debug:
        debug_info = {
            'coarse_mask': coarse_mask,
            'denoised_mask': denoised_mask,
            'skeleton': skeleton,
            'outer_region': outer_region,
            'dist_to_stroke': dist_to_stroke,
            'D_map': D_map,
            'mask_middle': (mask_outer & (~mask_skeleton)).astype(np.uint8) * 255
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

        # 坐标平移到 patch 局部空间，缩放到 [0, patch_size-1]
        pts[:, 0] -= crop_x1
        pts[:, 1] -= crop_y1
        # 边界 patch 的 crop 尺寸可能小于 patch_size（需 padding 补齐），须先缩放再 clip
        scale_x = patch_size / max(crop_x2 - crop_x1, 1)
        scale_y = patch_size / max(crop_y2 - crop_y1, 1)
        pts[:, 0] = pts[:, 0] * scale_x
        pts[:, 1] = pts[:, 1] * scale_y
        pts = np.clip(pts, 0, patch_size - 1).astype(np.int32)

        cv2.fillPoly(Mb, [pts], 1)

    return Mb
