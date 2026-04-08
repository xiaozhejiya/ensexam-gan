"""
色彩数据增强工具：

  recolor_stroke          : 将手写字迹替换为任意颜色
  colorize_printed_text   : 对 GT 中随机选取的部分印刷文字词组着色，Iin / Igt 同步更新
  recolor_stroke_and_tint : 组合增强（笔迹换色 + 部分印刷文字着色）

所有函数均对 numpy RGB uint8 图像操作，与训练流程解耦。
"""

import os
import random

import cv2
import numpy as np


# ── 公共工具 ────────────────────────────────────────────────────────────────────

def create_class_mask(txt_path: str,
                       img_h: int,
                       img_w: int,
                       target_class: int = 1) -> np.ndarray:
    """
    从 box_label_txt 标注文件生成指定类别的二值掩码（全图分辨率）。

    标注格式（每行）：x1,y1,x2,y2,x3,y3,x4,y4, class
    类别说明：1 = 手写字迹，2 = 批改痕迹

    Args:
        txt_path     : 对应图像的标注文件路径
        img_h/img_w  : 原图高宽
        target_class : 目标类别（1 或 2）

    Returns:
        mask : uint8 (img_h, img_w)，目标类别区域=1，其余=0
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(txt_path):
        return mask

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 9:
                continue
            try:
                cls = int(parts[8])
                if cls != target_class:
                    continue
                coords = [int(p) for p in parts[:8]]
            except ValueError:
                continue
            pts = np.array(coords, dtype=np.int32).reshape(4, 2)
            cv2.fillPoly(mask, [pts], 1)
    return mask


# ── 内部工具 ────────────────────────────────────────────────────────────────────

def _extract_stroke_alpha(Iin: np.ndarray,
                           Igt: np.ndarray,
                           threshold: int = 15,
                           norm_scale: float = 35.0,
                           region_mask: np.ndarray = None) -> np.ndarray:
    """
    从 (原图, GT) 对提取笔迹覆盖度图（软 alpha）。

    手写墨迹使像素变暗，故 Igt - Iin > 0 的区域为笔迹。
    JPEG 压缩噪点通过形态学开闭运算去除。

    Returns:
        alpha : float32 (H, W)，取值 [0, 1]，笔迹中心接近 1.0
    """
    diff     = Igt.astype(np.int16) - Iin.astype(np.int16)
    darkness = np.clip(diff.mean(axis=2), 0, None).astype(np.float32)

    binary = (darkness > threshold).astype(np.uint8)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 只做闭运算（连接断裂笔画），不做开运算
    # 开运算会删除宽度不足 3px 的细笔画，导致轻细笔迹被打断成离散点
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # 若提供区域掩码，只在该区域内计算 alpha（用于区分手写/批改类别）
    if region_mask is not None:
        binary = binary & (region_mask > 0).astype(np.uint8)

    alpha = np.where(binary > 0, np.clip(darkness / norm_scale, 0, 1), 0.0)
    return alpha.astype(np.float32)


def _random_vivid_color() -> tuple:
    """生成随机鲜艳颜色（HSV 高饱和度），返回 RGB tuple。"""
    h = random.randint(0, 179)          # 随机色相
    s = random.randint(180, 255)        # 高饱和度
    v = random.randint(160, 220)        # 中高亮度（避免过亮或过暗）
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8),
                        cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR → RGB


def _apply_text_color(Igt: np.ndarray,
                       char_mask: np.ndarray,
                       gray: np.ndarray,
                       color: tuple) -> np.ndarray:
    """
    对 char_mask 为 True 的文字像素做颜色替换。

    原理：
        intensity = (255 - gray) / 255   # 越黑越接近 1
        new_pixel = 255 × (1-intensity) + target_color × intensity
        → 纯黑文字变为 target_color，灰色文字变为浅色，白色背景不动
    """
    result    = Igt.astype(np.float32).copy()
    intensity = (255.0 - gray.astype(np.float32)) / 255.0   # (H, W)

    for c, tc in enumerate(color):
        ch = result[:, :, c]
        ch[char_mask] = 255.0 + (tc - 255.0) * intensity[char_mask]
        result[:, :, c] = ch

    return np.clip(result, 0, 255).astype(np.uint8)


# ── 公开接口 ────────────────────────────────────────────────────────────────────

def recolor_stroke(Iin: np.ndarray,
                   Igt: np.ndarray,
                   target_color: tuple,
                   threshold: int = 15,
                   norm_scale: float = 60.0,
                   class1_mask: np.ndarray = None,
                   class2_mask: np.ndarray = None) -> np.ndarray:
    """
    将手写字迹（类别 1）颜色替换为 target_color，批改痕迹（类别 2）严格保留。

    Args:
        Iin          : 含笔迹原图，RGB uint8 (H, W, 3)
        Igt          : 干净底图（GT），RGB uint8 (H, W, 3)
        target_color : 目标墨迹颜色，RGB tuple (0~255)
        threshold    : 笔迹检测阈值（建议 10~20）
        norm_scale   : alpha 归一化系数；调大→笔迹颜色更淡，调小→更饱和
        class1_mask  : 手写字迹区域掩码 uint8 (H, W)
        class2_mask  : 批改痕迹区域掩码 uint8 (H, W)；显式排除，避免因
                       class1 标注框包含 class2 区域而误覆盖批改痕迹

    Returns:
        Iin_new : 仅 class1 区域换色、class2 区域严格保留的图像
    """
    # class2 从有效区域中显式剔除，防止标注框重叠导致误覆盖
    if class1_mask is not None and class2_mask is not None:
        effective_mask = (class1_mask & ~class2_mask).astype(np.uint8)
    else:
        effective_mask = class1_mask   # None 或只有其中一个时退化为原逻辑

    alpha  = _extract_stroke_alpha(Iin, Igt, threshold, norm_scale,
                                    region_mask=effective_mask)
    alpha3 = alpha[:, :, None]
    color  = np.array(target_color, dtype=np.float32)
    result = Igt.astype(np.float32) * (1 - alpha3) + color * alpha3

    # effective_mask 以外（背景 + class2）全部还原为原始 Iin
    if effective_mask is not None:
        outside = (effective_mask == 0)[:, :, None]
        result  = np.where(outside, Iin.astype(np.float32), result)
    return np.clip(result, 0, 255).astype(np.uint8)


def colorize_printed_text(Iin: np.ndarray,
                           Igt: np.ndarray,
                           color_ratio: float = 0.25,
                           n_colors: int = 2,
                           colors: list = None,
                           dilation_px: int = 15,
                           min_area: int = 300,
                           text_threshold: int = 180,
                           stroke_threshold: int = 15,
                           stroke_norm_scale: float = 60.0) -> tuple:
    """
    对 GT 中随机选取的部分印刷文字词组进行着色，Iin / Igt 同步更新。
    手写笔迹区域在 Iin 中保持原始颜色不受影响。

    实现步骤：
        1. Igt 灰度二值化 → 提取字符级连通域
        2. 对字符掩码做矩形膨胀 → 让同词/同行字符聚合成词组
        3. 在词组级连通域上随机采样 color_ratio 比例
        4. 对选中词组的原始字符像素做颜色替换
        5. Iin 的背景区域同步更新，笔迹区域不变

    Args:
        Iin              : 含笔迹原图，RGB uint8 (H, W, 3)
        Igt              : 干净底图，  RGB uint8 (H, W, 3)
        color_ratio      : 被着色的词组比例 [0, 1]，建议 0.1~0.4
        n_colors         : 同时使用的颜色数量，建议 1~3
        colors           : 指定颜色列表（RGB tuple），None = 全部随机生成
        dilation_px      : 词组聚合时的水平膨胀宽度（像素）
                           图像分辨率越高，该值应越大（512px图建议10~15，
                           全图2000px+建议40~60）
        min_area         : 词组最小面积（像素²），过滤标点/噪点，建议200~500
        text_threshold   : 判定为文字像素的灰度上限（0~255），越大覆盖越多
        stroke_threshold : 笔迹检测阈值，同 recolor_stroke
        stroke_norm_scale: 笔迹 alpha 归一化系数，同 recolor_stroke

    Returns:
        (Iin_new, Igt_new) : 增强后的图像对，均为 RGB uint8 (H, W, 3)
    """
    # ── 1. 提取字符掩码 ────────────────────────────────────────────────────────
    gray = cv2.cvtColor(Igt, cv2.COLOR_RGB2GRAY)
    _, char_binary = cv2.threshold(gray, text_threshold, 255, cv2.THRESH_BINARY_INV)
    # char_binary: 文字像素=255，背景=0

    # ── 2. 水平膨胀聚合成词组 ──────────────────────────────────────────────────
    # 宽核横向延伸让同单词的字母相连，高度较小避免跨行合并
    kw = max(dilation_px, 1)
    kh = max(dilation_px // 4, 1)
    k_word = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(char_binary, k_word, iterations=1)

    # ── 3. 词组级连通域分析 ────────────────────────────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8)

    valid_labels = [i for i in range(1, n_labels)
                    if stats[i, cv2.CC_STAT_AREA] >= min_area]

    if not valid_labels:
        return Iin.copy(), Igt.copy()

    # ── 4. 随机采样 & 分配颜色 ────────────────────────────────────────────────
    n_select = max(1, int(len(valid_labels) * color_ratio))
    selected = random.sample(valid_labels, min(n_select, len(valid_labels)))

    if colors is None:
        palette = [_random_vivid_color() for _ in range(n_colors)]
    else:
        palette = list(colors)

    label_color = {lbl: palette[i % len(palette)]
                   for i, lbl in enumerate(selected)}

    # ── 5. 对选中词组的字符像素着色 ────────────────────────────────────────────
    Igt_new = Igt.copy()
    for lbl, color in label_color.items():
        word_region = (labels == lbl)               # 膨胀后的词组区域
        pixel_mask  = word_region & (char_binary > 0)  # 仅真实文字像素
        if not pixel_mask.any():
            continue
        Igt_new = _apply_text_color(Igt_new, pixel_mask, gray, color)

    # ── 6. Iin 背景同步更新，笔迹区域保持原色 ──────────────────────────────────
    stroke_alpha = _extract_stroke_alpha(
        Iin, Igt, stroke_threshold, stroke_norm_scale)
    alpha3  = stroke_alpha[:, :, None]
    Iin_new = (Iin.astype(np.float32) * alpha3
               + Igt_new.astype(np.float32) * (1 - alpha3))

    return np.clip(Iin_new, 0, 255).astype(np.uint8), Igt_new


def recolor_stroke_and_tint(Iin: np.ndarray,
                             Igt: np.ndarray,
                             stroke_color: tuple = None,
                             color_ratio: float = 0.25,
                             n_colors: int = 2,
                             dilation_px: int = 15,
                             min_area: int = 300,
                             text_threshold: int = 180,
                             threshold: int = 15,
                             norm_scale: float = 60.0,
                             class1_mask: np.ndarray = None,
                             class2_mask: np.ndarray = None) -> tuple:
    """
    组合增强：先对部分印刷文字着色，再替换手写笔迹颜色。

    Args:
        stroke_color : 笔迹目标颜色，None = 随机生成
        其余参数     : 同 colorize_printed_text / recolor_stroke

    Returns:
        (Iin_new, Igt_new)
    """
    if stroke_color is None:
        stroke_color = _random_vivid_color()

    # 先对底图着色
    Iin_mid, Igt_new = colorize_printed_text(
        Iin, Igt,
        color_ratio=color_ratio, n_colors=n_colors,
        dilation_px=dilation_px, min_area=min_area,
        text_threshold=text_threshold,
        stroke_threshold=threshold, stroke_norm_scale=norm_scale,
    )
    # 再替换笔迹颜色（基于着色后的 Igt_new 作为背景参考，仅作用于 class1 区域）
    Iin_new = recolor_stroke(Iin_mid, Igt_new, stroke_color,
                              threshold, norm_scale,
                              class1_mask=class1_mask,
                              class2_mask=class2_mask)
    return Iin_new, Igt_new
