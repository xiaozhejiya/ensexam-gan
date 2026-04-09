"""
字迹插入增强工具：

  insert_strokes : 在完整图像上提取纯净笔迹差分，插入到 Igt 的空白区域，仅修改 Iin。

核心原理：
  diff = clip(Igt − Iin, 0, 255)
  印刷题目在 Iin 与 Igt 中完全相同，相减归零；差分中只剩墨迹使像素变暗的贡献。
  因此 diff_patch 天然不含任何印刷内容，无需来源背景过滤或复杂 alpha 替换。

插入公式：
  Iin_new[dst] = clip(Iin[dst] − diff_patch, 0, 255)
  等价于把墨迹的"变暗量"叠加到目标区域，背景颜色由目标区域自身决定。

推荐在完整图像上调用（空白区域搜索范围更大，成功率更高）。
"""

import random

import cv2
import numpy as np

from tools.color_augment import _extract_stroke_alpha


# ── 内部工具 ────────────────────────────────────────────────────────────────────

def _build_content_mask(Igt: np.ndarray, text_threshold: int = 200) -> np.ndarray:
    """
    在 Igt 上标记"有内容"区域：灰度 < text_threshold 视为印刷文字。

    Returns:
        content_mask : uint8 (H, W)，有内容=1，空白=0
    """
    gray = cv2.cvtColor(Igt, cv2.COLOR_RGB2GRAY)
    return (gray < text_threshold).astype(np.uint8)


def _find_blank_positions(Igt: np.ndarray,
                           patch_h: int,
                           patch_w: int,
                           content_mask: np.ndarray,
                           text_threshold: int = 200,
                           blank_ratio: float = 0.97,
                           max_overlap: float = 0.03,
                           n_candidates: int = 500) -> list:
    """
    在 Igt 中随机采样，返回可放置 (patch_h × patch_w) 笔迹的空白起点列表。

    双重判定：
      1. 该区域在 Igt 上灰度 ≥ text_threshold 的比例 ≥ blank_ratio（确保无印刷题目）
      2. 与已有内容的重叠比例 ≤ max_overlap

    Returns:
        list of (y, x) 起点坐标
    """
    H, W = Igt.shape[:2]
    if H < patch_h or W < patch_w:
        return []

    gray = cv2.cvtColor(Igt, cv2.COLOR_RGB2GRAY)
    candidates = []

    for _ in range(n_candidates):
        y = random.randint(0, H - patch_h)
        x = random.randint(0, W - patch_w)

        region_gray    = gray[y:y+patch_h, x:x+patch_w]
        region_content = content_mask[y:y+patch_h, x:x+patch_w]

        if (region_gray >= text_threshold).mean() < blank_ratio:
            continue
        if region_content.mean() > max_overlap:
            continue

        candidates.append((y, x))

    return candidates


def _extract_diff_patches(diff: np.ndarray,
                            labels: np.ndarray,
                            stats: np.ndarray,
                            valid_labels: list,
                            n_patches: int,
                            pad: int = 8,
                            min_peak: int = 30) -> list:
    """
    从连通区域列表中提取差分 patch（Igt − Iin 的类别 1 区域切片）。

    Args:
        diff       : uint8 (H, W, 3)，clip(Igt − Iin, 0, 255)，已限制在 effective_mask 内
        min_peak   : diff 均值通道峰值下限，过低说明笔迹太浅，跳过（建议 20~50）

    Returns:
        list of dict: {'diff': uint8 (ph, pw, 3)}
    """
    H, W = diff.shape[:2]
    shuffled = list(valid_labels)
    random.shuffle(shuffled)

    patches = []
    for lbl in shuffled:
        if len(patches) >= n_patches:
            break

        x0 = stats[lbl, cv2.CC_STAT_LEFT]
        y0 = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]

        x1 = max(0, x0 - pad)
        y1 = max(0, y0 - pad)
        x2 = min(W, x0 + bw + pad)
        y2 = min(H, y0 + bh + pad)

        diff_p = diff[y1:y2, x1:x2]
        if diff_p.max() < min_peak:   # 笔迹变暗量太小，跳过
            continue

        patches.append({'diff': diff_p.copy()})

    return patches


# ── 公开接口 ────────────────────────────────────────────────────────────────────

def insert_strokes(Iin: np.ndarray,
                   Igt: np.ndarray,
                   class1_mask: np.ndarray = None,
                   class2_mask: np.ndarray = None,
                   n_insert: int = 5,
                   noise_threshold: int = 30,
                   min_patch_peak: int = 60,
                   min_area: int = 500,
                   text_threshold: int = 210,
                   return_positions: bool = False):
    """
    从 class1 区域提取纯净笔迹差分，插入到 Igt 的空白区域，仅修改 Iin。

    插入公式：
        Iin_new[dst] = clip(Iin[dst] − diff_patch, 0, 255)

    diff_patch = clip(Igt − Iin, 0, 255) 只含墨迹变暗贡献，
    印刷题目在 Iin/Igt 中相同，相减归零，天然不会被搬运。

    Args:
        Iin             : 含笔迹原图，RGB uint8 (H, W, 3)
        Igt             : 干净底图，  RGB uint8 (H, W, 3)
        class1_mask     : 手写字迹区域掩码 uint8 (H, W)
        class2_mask     : 批改痕迹区域掩码 uint8 (H, W)，显式排除
        n_insert        : 最多插入的笔迹 patch 数量（建议 3~8）
        noise_threshold : 噪声过滤阈值：逐像素 diff 均值 < 此值归零，保留原始值
                          控制"题目附近的 GT 修改噪声"被滤除的严格程度（建议 20~50）
        min_patch_peak  : patch 质量阈值：整块 patch 的最大 diff < 此值则跳过
                          控制"只挑选墨色足够深的笔迹 patch"（建议 50~100）
        min_area        : 连通区域最小面积（全图建议 300~800）
        text_threshold  : 目标空白判定阈值：GT 灰度低于此值视为有内容（建议 200~225）
        return_positions: 若为 True，额外返回 list of (y, x, ph, pw) 插入坐标

    Returns:
        Iin_new          : RGB uint8 (H, W, 3)
        positions (可选) : list of (y, x, ph, pw)
    """
    if class1_mask is not None and class2_mask is not None:
        effective_mask = (class1_mask & ~class2_mask).astype(np.uint8)
    else:
        effective_mask = class1_mask

    # ── 1. 计算差分并用二值掩码过滤噪声 ─────────────────────────────────────
    # 数据集作者在擦除字迹时，邻近印刷题目的像素可能被轻微影响，
    # 导致 Igt − Iin 在题目区域不严格为 0（通常为 10~50 的噪声）。
    #
    # 二值掩码：均值 diff < noise_threshold → 整个像素归零，否则保留原始差分值
    #   diff_mean < noise_threshold → 0（题目噪声完全归零，原值不保留）
    #   diff_mean ≥ noise_threshold → 保留原始 diff（笔迹颜色真实）
    diff_raw  = np.clip(
        Igt.astype(np.int16) - Iin.astype(np.int16), 0, 255
    ).astype(np.uint8)
    diff_mean = diff_raw.mean(axis=2)                               # (H, W)
    keep_mask = (diff_mean >= noise_threshold).astype(np.uint8)    # 1=笔迹, 0=噪声
    diff_full = diff_raw * keep_mask[:, :, None]                    # 保留原值，不做减法

    # 限制在 class1（去除 class2）区域内
    if effective_mask is not None:
        diff_full = diff_full * effective_mask[:, :, None]

    # ── 2. 连通域分析（基于阈值处理后的差分图） ────────────────────────────────
    diff_gray = diff_full.mean(axis=2).astype(np.uint8)
    binary    = (diff_gray > 0).astype(np.uint8)    # 已经过阈值，直接判断非零

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    valid_labels = [i for i in range(1, n_labels)
                    if stats[i, cv2.CC_STAT_AREA] >= min_area]

    if not valid_labels:
        return (Iin.copy(), []) if return_positions else Iin.copy()

    # ── 3. 提取差分 patch（只取墨色足够深的 patch） ───────────────────────────
    patches = _extract_diff_patches(
        diff_full, labels, stats, valid_labels, n_insert * 3,
        min_peak=min_patch_peak)
    if not patches:
        return (Iin.copy(), []) if return_positions else Iin.copy()

    # ── 4. 构建已有内容掩码（印刷文字 + 已有笔迹），插入时双重规避 ──────────────
    content_mask = _build_content_mask(Igt, text_threshold)
    content_mask = np.maximum(content_mask, binary)

    # ── 5. 寻找空白位置并逐 patch 插入 ────────────────────────────────────────
    result    = Iin.astype(np.int16).copy()
    positions = []
    inserted  = 0

    for patch in patches:
        if inserted >= n_insert:
            break

        ph, pw = patch['diff'].shape[:2]
        pos_list = _find_blank_positions(
            Igt, ph, pw, content_mask,
            text_threshold=text_threshold)

        if not pos_list:
            continue

        y, x = random.choice(pos_list)

        # 将墨迹"变暗量"叠加到目标区域（直接相减，目标背景保持原色）
        result[y:y+ph, x:x+pw] -= patch['diff'].astype(np.int16)

        # 更新内容掩码，防止后续 patch 重叠
        new_stroke = (patch['diff'].mean(axis=2) > noise_threshold).astype(np.uint8)
        content_mask[y:y+ph, x:x+pw] = np.maximum(
            content_mask[y:y+ph, x:x+pw], new_stroke)

        positions.append((y, x, ph, pw))
        inserted += 1

    Iin_new = np.clip(result, 0, 255).astype(np.uint8)
    return (Iin_new, positions) if return_positions else Iin_new
