"""
字迹插入增强工具，提供两种来源的笔迹插入方式：

┌─────────────────────┬──────────────────────────────────────────────────────┐
│ 函数                │ 说明                                                 │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ insert_strokes      │ 从考卷自身提取笔迹（exam 模式）                      │
│                     │ 原理：diff = clip(Igt − Iin, 0, 255)                │
│                     │ 印刷题目在 Iin/Igt 中相同，相减归零，只剩墨迹贡献。 │
│                     │ 需要 noise_threshold 过滤 GT 标注噪声。              │
│                     │ 优点：风格与原卷完全一致；缺点：受数据集标注质量影响│
├─────────────────────┼──────────────────────────────────────────────────────┤
│ insert_strokes_     │ 从外部笔迹库加载（library 模式）                     │
│ from_library        │ 原理：diff = clip(255 − 白底手写页面, 0, 255)        │
│                     │ 背景纯白，diff 在背景处严格为 0，无题目噪声问题。    │
│                     │ 支持缩放（scale_range）和变色（ink_color）增强。     │
│                     │ 优点：纯净无噪声；缺点：需提前用 build_stroke_       │
│                     │       library.py 准备笔迹库                         │
└─────────────────────┴──────────────────────────────────────────────────────┘

两种模式插入公式相同：
  Iin_new[dst] = clip(Iin[dst] − diff_patch, 0, 255)
"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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


# ── 笔迹库增强工具 ──────────────────────────────────────────────────────────────

def _recolor_diff(diff: np.ndarray, ink_color: tuple) -> np.ndarray:
    """
    将灰色笔迹 diff patch 重染为指定墨水颜色。

    原理：
        diff 的每个像素代表"比白色暗多少"（0=背景，255=最深墨迹）。
        归一化为透明度 alpha = diff_gray / 255，再按目标墨水颜色重建各通道：
            diff_new[c] = (255 - ink_color[c]) * alpha

    Args:
        diff      : RGB uint8 (H, W, 3)，高值=深墨迹
        ink_color : (R, G, B) 目标墨水颜色，如 (30, 60, 150) 深蓝

    Returns:
        RGB uint8 (H, W, 3)，重染后的 diff
    """
    alpha = diff.mean(axis=2) / 255.0                          # (H, W) 透明度
    r, g, b = ink_color
    diff_colored = np.stack([
        (255 - r) * alpha,
        (255 - g) * alpha,
        (255 - b) * alpha,
    ], axis=2)
    return np.clip(diff_colored, 0, 255).astype(np.uint8)


def _scale_diff(diff: np.ndarray, scale: float) -> np.ndarray:
    """
    对 diff patch 进行缩放。

    Args:
        diff  : RGB uint8 (H, W, 3)
        scale : 缩放比例，如 0.8 缩小，1.2 放大

    Returns:
        RGB uint8，缩放后的 diff
    """
    h, w = diff.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(diff, (new_w, new_h), interpolation=interp)


def _random_ink_color() -> tuple:
    """
    随机生成笔迹颜色，兼顾考试常见用笔与增强多样性。

    采样策略（参考 color_augment._random_vivid_color）：
        50% 考试常见色：黑色系 / 深蓝 / 蓝色
        50% 全色相随机：HSV 高饱和度随机采样，覆盖红/绿/紫等非常规颜色

    Returns:
        (R, G, B) tuple
    """
    if random.random() < 0.5:
        # ── 考试常见色 ──
        mode = random.random()
        if mode < 0.4:
            v = random.randint(10, 40)
            return (v, v, v)                           # 黑色系
        elif mode < 0.75:
            return (random.randint(10, 50),
                    random.randint(20, 80),
                    random.randint(100, 180))           # 深蓝/蓝黑
        else:
            return (random.randint(30, 80),
                    random.randint(80, 140),
                    random.randint(180, 230))           # 蓝色
    else:
        # ── 全色相随机（高饱和度），同 _random_vivid_color ──
        h = random.randint(0, 179)
        s = random.randint(180, 255)
        v = random.randint(80, 200)                    # 适当压暗，模拟真实墨迹
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8),
                            cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


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
                   margin: int = 30,
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
        margin          : 内容区域外扩禁区（像素），确保插入字迹与题目/已有笔迹保持距离
                          （建议 20~50）
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
    # 向外膨胀 margin 像素，保证插入字迹与题目/已有笔迹保持安全距离
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (margin * 2 + 1, margin * 2 + 1))
        content_mask = cv2.dilate(content_mask, k)

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

        # 更新内容掩码，尽量避免后续 patch 与当前 patch 重叠。
        # 注：_find_blank_positions 基于随机采样而非穷举，极少数情况下仍可能出现
        # 轻微重叠，属于正常现象——实际考卷中学生字迹本身也存在笔画交叠。
        new_stroke = (patch['diff'].mean(axis=2) > noise_threshold).astype(np.uint8)
        content_mask[y:y+ph, x:x+pw] = np.maximum(
            content_mask[y:y+ph, x:x+pw], new_stroke)

        positions.append((y, x, ph, pw))
        inserted += 1

    Iin_new = np.clip(result, 0, 255).astype(np.uint8)
    return (Iin_new, positions) if return_positions else Iin_new


def insert_strokes_from_library(
        Iin: np.ndarray,
        Igt: np.ndarray,
        library_dir: str,
        n_insert: int = 5,
        scale_range: tuple = (0.7, 1.3),
        ink_color = 'random',
        text_threshold: int = 210,
        margin: int = 30,
        return_positions: bool = False):
    """
    从笔迹库（build_stroke_library.py 生成的 patches/ 目录）随机抽取 diff patch，
    施加缩放和变色增强后插入到 Igt 的空白区域，仅修改 Iin。

    插入公式：
        Iin_new[dst] = clip(Iin[dst] − diff_patch, 0, 255)

    与 insert_strokes 相比，此函数使用外部纯净笔迹库，
    完全避免考卷差分中题目噪声的干扰，无需 noise_threshold / min_patch_peak。

    Args:
        Iin            : 含笔迹原图，RGB uint8 (H, W, 3)
        Igt            : 干净底图，  RGB uint8 (H, W, 3)
        library_dir    : 笔迹库 patches 目录路径（含 *.png diff 文件）
        n_insert       : 最多插入的笔迹数量（建议 3~8）
        scale_range    : 缩放范围 (min_scale, max_scale)，如 (0.7, 1.3)
        ink_color      : 墨水颜色，支持：
                           'random' → 每次随机采样考试场景颜色（黑/深蓝/蓝）
                           (R, G, B) → 固定颜色
                           None     → 保留 patch 原始颜色，不重染
        text_threshold : 目标空白判定阈值（建议 200~225）
        margin         : 内容区域外扩禁区（像素），确保插入字迹与题目保持距离（建议 20~50）
        return_positions: 若为 True，额外返回 list of (y, x, ph, pw)

    Returns:
        Iin_new          : RGB uint8 (H, W, 3)
        positions (可选) : list of (y, x, ph, pw)
    """
    # ── 1. 加载笔迹库文件列表 ──────────────────────────────────────────────────
    lib_path = Path(library_dir)
    patch_files = sorted(lib_path.glob('*.png'))
    if not patch_files:
        return (Iin.copy(), []) if return_positions else Iin.copy()

    # ── 2. 随机抽取并施加增强 ─────────────────────────────────────────────────
    candidates = random.choices(patch_files, k=n_insert * 3)
    patches = []
    for fpath in candidates:
        diff = np.array(Image.open(fpath).convert('RGB'))

        # 缩放
        scale = random.uniform(*scale_range)
        diff  = _scale_diff(diff, scale)

        # 变色
        if ink_color == 'random':
            color = _random_ink_color()
            diff  = _recolor_diff(diff, color)
        elif ink_color is not None:
            diff = _recolor_diff(diff, ink_color)
        # ink_color=None 时保留原色

        patches.append(diff)

    # ── 3. 构建内容掩码（印刷文字区域），插入时规避 ───────────────────────────
    content_mask = _build_content_mask(Igt, text_threshold)
    # 向外膨胀 margin 像素，保证插入字迹与题目/已有笔迹保持安全距离
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (margin * 2 + 1, margin * 2 + 1))
        content_mask = cv2.dilate(content_mask, k)

    # ── 4. 寻找空白位置并逐 patch 插入 ────────────────────────────────────────
    result    = Iin.astype(np.int16).copy()
    positions = []
    inserted  = 0

    for diff in patches:
        if inserted >= n_insert:
            break

        ph, pw = diff.shape[:2]
        pos_list = _find_blank_positions(
            Igt, ph, pw, content_mask,
            text_threshold=text_threshold)

        if not pos_list:
            continue

        y, x = random.choice(pos_list)
        result[y:y+ph, x:x+pw] -= diff.astype(np.int16)

        # 更新内容掩码，尽量避免后续 patch 与当前 patch 重叠。
        # 注：极少数情况下仍可能出现轻微重叠，属于正常现象——
        # 实际考卷中学生字迹本身也存在笔画交叠。
        new_stroke = (diff.mean(axis=2) > 10).astype(np.uint8)
        content_mask[y:y+ph, x:x+pw] = np.maximum(
            content_mask[y:y+ph, x:x+pw], new_stroke)

        positions.append((y, x, ph, pw))
        inserted += 1

    Iin_new = np.clip(result, 0, 255).astype(np.uint8)
    return (Iin_new, positions) if return_positions else Iin_new
