"""
字迹插入增强可视化（全图级）：在完整图像上运行插入增强，展示前后对比。

用法：
    python tools/visualize_stroke_insert.py
    python tools/visualize_stroke_insert.py --n_samples 3
    python tools/visualize_stroke_insert.py --files 268.jpg 87.jpg --save_path output/insert.png

输出两张图：
    <save_path>           : 全图对比（原图 | 插入后+绿框），每行一个样本
    <save_path>.crops.png : 各插入区域放大对比（原 | 插入后）
"""

import argparse
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

for _fn in ['Microsoft YaHei', 'SimHei', 'STHeiti', 'WenQuanYi Micro Hei']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _fn
        break
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.color_augment import create_class_mask
from tools.stroke_insert import insert_strokes

# ── 数据集路径 ──────────────────────────────────────────────────────────────────
DATA_ROOT = r"E:\dataset\SCUT-EnsExam"
SPLIT     = "train"
SEED      = 41

# ── 插入参数 ────────────────────────────────────────────────────────────────────
INSERT_PARAMS = dict(
    n_insert        = 10,  # 最多插入多少个笔迹 patch（建议 3~8）
    noise_threshold = 50,  # 噪声过滤阈值：逐像素 diff 均值 < 此值归零（建议 20~50）
                           # 控制"题目附近 GT 修改噪声"被滤除的严格程度
    min_patch_peak  = 50,  # patch 质量阈值：整块 patch 最大 diff < 此值则跳过（建议 50~100）
                           # 控制"只挑选墨色足够深的笔迹 patch"
    min_area        = 500, # 最小连通区域面积，全图建议 300~800
    text_threshold  = 210, # 目标空白判定灰度阈值（建议 200~225）
                           # 调大→空白判定宽松（易找位置，可能插到浅色题目）
                           # 调小→严格（安全，但可能找不到位置）
)

# ── 可视化参数 ──────────────────────────────────────────────────────────────────
DISPLAY_MAX_H  = 1600   # 全图缩略图最大高度（像素），决定主图清晰度
ZOOM_PAD       = 40     # 放大裁剪时额外保留的边距
ZOOM_SIZE      = 300    # 每张放大图的显示尺寸（像素，正方形）
ZOOM_PER_ROW   = 4      # 放大图每行显示几对
BOX_COLOR      = (0, 220, 80)
BOX_THICKNESS  = 6


# ── 工具函数 ────────────────────────────────────────────────────────────────────

def scale_to_max_h(img: np.ndarray, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    return cv2.resize(img, (int(w * max_h / h), max_h), interpolation=cv2.INTER_AREA)


def draw_boxes(img: np.ndarray, positions: list, scale: float) -> np.ndarray:
    vis = img.copy()
    for y, x, ph, pw in positions:
        pt1 = (int(x * scale), int(y * scale))
        pt2 = (int((x + pw) * scale), int((y + ph) * scale))
        cv2.rectangle(vis, pt1, pt2, BOX_COLOR, BOX_THICKNESS)
    return vis


def extract_zoom_crop(img: np.ndarray, y, x, ph, pw) -> np.ndarray:
    H, W = img.shape[:2]
    y1 = max(0, y - ZOOM_PAD)
    x1 = max(0, x - ZOOM_PAD)
    y2 = min(H, y + ph + ZOOM_PAD)
    x2 = min(W, x + pw + ZOOM_PAD)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_AREA)


# ── 主函数 ──────────────────────────────────────────────────────────────────────

def main(n_samples=3, save_path=None, seed=SEED, files=None):
    random.seed(seed)
    np.random.seed(seed)

    img_dir = os.path.join(DATA_ROOT, SPLIT, "all_images")
    gt_dir  = os.path.join(DATA_ROOT, SPLIT, "all_labels")
    txt_dir = os.path.join(DATA_ROOT, SPLIT, "box_label_txt")

    if files:
        candidate_files = [f if f.endswith(('.jpg', '.png', '.jpeg')) else f + '.jpg'
                           for f in files]
    else:
        candidate_files = sorted([f for f in os.listdir(img_dir)
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        random.shuffle(candidate_files)

    # ── 收集样本 & 运行增强 ────────────────────────────────────────────────────
    results = []
    for fname in candidate_files:
        if not files and len(results) >= n_samples:
            break
        img_path = os.path.join(img_dir, fname)
        gt_path  = os.path.join(gt_dir,  fname)
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            print(f"  ✗ {fname} 不存在，跳过")
            continue

        Iin = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        Igt = cv2.cvtColor(cv2.imread(gt_path),  cv2.COLOR_BGR2RGB)
        H, W = Iin.shape[:2]

        txt_path    = os.path.join(txt_dir, os.path.splitext(fname)[0] + '.txt')
        class1_mask = create_class_mask(txt_path, H, W, target_class=1)
        class2_mask = create_class_mask(txt_path, H, W, target_class=2)

        Iin_aug, positions = insert_strokes(
            Iin, Igt,
            class1_mask=class1_mask,
            class2_mask=class2_mask,
            return_positions=True,
            **INSERT_PARAMS,
        )

        # 差分图：与 insert_strokes 内部完全一致（二值掩码，用 noise_threshold）
        #   diff_mean < noise_threshold → 0；diff_mean ≥ noise_threshold → 保留原值
        # 显示时取反（255 − diff）：笔迹呈深色，背景/题目区域呈白色
        thr      = INSERT_PARAMS['noise_threshold']
        diff_raw = np.clip(
            Igt.astype(np.int16) - Iin.astype(np.int16), 0, 255
        ).astype(np.uint8)
        keep_mask  = (diff_raw.mean(axis=2) >= thr).astype(np.uint8)
        diff_clean = diff_raw * keep_mask[:, :, None]
        diff_vis   = (255 - diff_clean).astype(np.uint8)  # 深笔迹 / 白背景

        print(f"  ✓ {fname}  {H}×{W}  插入 {len(positions)} 处")
        results.append(dict(fname=fname, Iin=Iin, Igt=Igt,
                            diff_vis=diff_vis,
                            Iin_aug=Iin_aug, positions=positions))

    if not results:
        print("未处理任何图像，请检查数据集路径。")
        return

    out_full  = save_path or "output/stroke_insert_preview.png"
    out_crops = out_full.replace('.png', '.crops.png')
    os.makedirs(os.path.dirname(os.path.abspath(out_full)), exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 图 1：全图对比（每行一个样本，两列：原图 | 插入后）
    # ══════════════════════════════════════════════════════════════════════════
    n = len(results)

    # 估算每个样本缩略图的宽高比
    sample_imgs = []
    for data in results:
        H, W = data['Iin'].shape[:2]
        scale = min(DISPLAY_MAX_H / H, 1.0)
        th = int(H * scale)
        tw = int(W * scale)
        thumb_orig = cv2.resize(data['Iin'],      (tw, th), interpolation=cv2.INTER_AREA)
        thumb_diff = cv2.resize(data['diff_vis'], (tw, th), interpolation=cv2.INTER_AREA)
        thumb_aug  = cv2.resize(data['Iin_aug'],  (tw, th), interpolation=cv2.INTER_AREA)
        thumb_aug  = draw_boxes(thumb_aug, data['positions'], scale)
        sample_imgs.append((thumb_orig, thumb_diff, thumb_aug, th, tw))

    # 所有样本统一用最大宽度
    max_tw = max(tw for *_, tw in sample_imgs)
    max_th = max(th for *_, th, _ in sample_imgs)

    # figsize：宽度容纳 3 张图，高度容纳 n 行
    dpi      = 100
    col_inch = max_tw / dpi
    row_inch = max_th / dpi
    fig_w    = col_inch * 3 + 0.6   # 3 列 + 间距
    fig_h    = row_inch * n + 0.4 * n + 0.3

    fig1, axes1 = plt.subplots(n, 3, figsize=(fig_w, fig_h), dpi=dpi)
    if n == 1:
        axes1 = [axes1]

    for r, (data, (thumb_orig, thumb_diff, thumb_aug, th, tw)) in enumerate(zip(results, sample_imgs)):
        axes1[r][0].imshow(thumb_orig)
        axes1[r][0].set_title(f"{data['fname']}  原图 Iin",
                               fontsize=9, fontweight='bold', pad=4)
        axes1[r][0].axis('off')

        axes1[r][1].imshow(thumb_diff)
        axes1[r][1].set_title("差分图（255 − clip(Igt−Iin, 0, 255)）\n深色=笔迹，白色=背景/印刷题目",
                               fontsize=8, fontweight='bold', pad=4)
        axes1[r][1].axis('off')

        axes1[r][2].imshow(thumb_aug)
        axes1[r][2].set_title(
            f"插入后 Iin（绿框 = 插入位置，共 {len(data['positions'])} 处）",
            fontsize=9, fontweight='bold', pad=4)
        axes1[r][2].axis('off')

    fig1.text(0.01, 0.002,
              f"n_insert={INSERT_PARAMS['n_insert']}  "
              f"noise_threshold={INSERT_PARAMS['noise_threshold']}  "
              f"min_patch_peak={INSERT_PARAMS['min_patch_peak']}  "
              f"min_area={INSERT_PARAMS['min_area']}  "
              f"text_threshold={INSERT_PARAMS['text_threshold']}",
              fontsize=7, color='#666')
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    fig1.savefig(out_full, bbox_inches='tight')
    print(f"全图对比已保存：{out_full}")
    plt.close(fig1)

    # ══════════════════════════════════════════════════════════════════════════
    # 图 2：放大对比（每行 ZOOM_PER_ROW 对，原图|插入后交替）
    # ══════════════════════════════════════════════════════════════════════════
    all_crops = []  # list of (label, crop_orig, crop_aug)
    for data in results:
        for k, (y, x, ph, pw) in enumerate(data['positions']):
            label     = f"{data['fname']}  #{k+1}"
            crop_orig = extract_zoom_crop(data['Iin'],     y, x, ph, pw)
            crop_aug  = extract_zoom_crop(data['Iin_aug'], y, x, ph, pw)
            all_crops.append((label, crop_orig, crop_aug))

    if all_crops:
        n_pairs  = len(all_crops)
        n_cols   = ZOOM_PER_ROW * 2       # 每行：orig aug | orig aug | ...
        n_rows   = (n_pairs + ZOOM_PER_ROW - 1) // ZOOM_PER_ROW

        cell_inch = ZOOM_SIZE / dpi
        fig2, axes2 = plt.subplots(
            n_rows, n_cols,
            figsize=(cell_inch * n_cols + 0.2, cell_inch * n_rows + 0.5 * n_rows),
            dpi=dpi)

        # 统一为 2D 数组
        if n_rows == 1:
            axes2 = [axes2]
        axes2 = [row if hasattr(row, '__len__') else [row] for row in axes2]

        for i, (label, crop_orig, crop_aug) in enumerate(all_crops):
            r   = i // ZOOM_PER_ROW
            c   = i %  ZOOM_PER_ROW
            ax_l = axes2[r][c * 2]
            ax_r = axes2[r][c * 2 + 1]

            ax_l.imshow(crop_orig)
            ax_l.set_title(f"{label}\n原图", fontsize=6.5, pad=2)
            ax_l.axis('off')

            ax_r.imshow(crop_aug)
            ax_r.set_title(f"{label}\n插入后", fontsize=6.5, pad=2)
            ax_r.axis('off')

        # 隐藏多余格子
        for i in range(len(all_crops), n_rows * ZOOM_PER_ROW):
            r = i // ZOOM_PER_ROW
            c = i %  ZOOM_PER_ROW
            axes2[r][c * 2].axis('off')
            axes2[r][c * 2 + 1].axis('off')

        plt.tight_layout()
        fig2.savefig(out_crops, bbox_inches='tight')
        print(f"放大对比已保存：{out_crops}")
        plt.close(fig2)
    else:
        print("所有图像均未找到合适空白区域，未生成放大对比图。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples',  type=int,  default=3)
    parser.add_argument('--save_path',  type=str,  default=None)
    parser.add_argument('--seed',       type=int,  default=SEED)
    parser.add_argument('--files',      type=str,  nargs='+', default=None)
    args = parser.parse_args()
    main(args.n_samples, args.save_path, args.seed, args.files)
