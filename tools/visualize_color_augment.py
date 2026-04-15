"""
色彩增强可视化：从数据集随机抽取图像，展示各类增强效果对比。

用法：
    python tools/visualize_color_augment.py
    python tools/visualize_color_augment.py --n_samples 4 --patch_size 512
    python tools/visualize_color_augment.py --save_path output/color_aug.png

每行一个样本，共 7 列：
    原图Iin | 原图GT | 笔迹换色Iin | 部分文字着色GT | 部分文字着色Iin | 组合增强Iin | 组合增强GT
"""

import argparse
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use('Agg')          # 无 GUI 环境也能保存图像
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 自动选择系统中可用的中文字体（Windows 优先 Microsoft YaHei / SimHei）
for _fn in ['Microsoft YaHei', 'SimHei', 'STHeiti', 'WenQuanYi Micro Hei']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = _fn
        break
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.color_augment import (recolor_stroke,
                                  colorize_printed_text,
                                  recolor_stroke_and_tint,
                                  create_class_mask,
                                  _random_vivid_color)
from utils.path_utils import normalize_path

# ── 数据集路径 ──────────────────────────────────────────────────────────────────
DATA_ROOT = normalize_path("~/dataset/SCUT-EnsExam")
SPLIT     = "train"
SEED      = 42

# ── 手写字迹换色参数 ────────────────────────────────────────────────────────────
STROKE_PARAMS = dict(
    threshold   = 15,    # 像素差阈值：低于此值的差异视为 JPEG 噪点而非笔迹（建议 10~20）
    norm_scale  = 100.0,  # alpha 饱和度：调小→笔画变粗/颜色更饱和，调大→笔画变细/颜色更淡
                         
)

# ── 印刷文字着色参数 ────────────────────────────────────────────────────────────
AUG_PARAMS = dict(
    color_ratio    = 0.25,   # 被着色的词组比例（建议 0.1~0.5）
    n_colors       = 3,      # 同时出现的颜色数（建议 1~3）
    dilation_px    = 15,     # 词组聚合膨胀宽度（像素，适用于 512px patch，建议 10~20）
    min_area       = 300,    # 词组最小面积，过滤标点/噪点（建议 200~600）
    text_threshold = 180,    # GT 灰度上限：低于此值才算印刷文字像素（建议 160~220）
)

# ── 工具函数 ────────────────────────────────────────────────────────────────────

def find_patch_with_strokes(img_path: str,
                             gt_path:  str,
                             patch_size: int,
                             n_candidates: int = 80,
                             min_stroke_ratio: float = 0.025) -> tuple:
    """
    在图像中随机采样，返回笔迹比例最高的 patch。
    返回 (Iin_patch, Igt_patch, crop_x, crop_y) 或 None。
    """
    Iin_full = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    Igt_full = cv2.cvtColor(cv2.imread(gt_path),  cv2.COLOR_BGR2RGB)
    H, W = Iin_full.shape[:2]
    if H < patch_size or W < patch_size:
        return None

    best_ratio, best = 0, None
    for _ in range(n_candidates):
        y = random.randint(0, H - patch_size)
        x = random.randint(0, W - patch_size)
        Iin_p = Iin_full[y:y+patch_size, x:x+patch_size]
        Igt_p = Igt_full[y:y+patch_size, x:x+patch_size]
        diff  = np.abs(Iin_p.astype(np.int16) - Igt_p.astype(np.int16)).mean(axis=2)
        ratio = (diff > 15).mean()
        if ratio > best_ratio:
            best_ratio, best = ratio, (Iin_p, Igt_p, x, y)

    if best_ratio < min_stroke_ratio:
        return None
    return best


def add_color_bar(ax, color_rgb, label):
    """在坐标轴下方添加小色块，说明当前笔迹/文字颜色。"""
    bar = np.ones((12, 60, 3), dtype=np.uint8)
    bar[:] = color_rgb
    ax.images[0]  # 确保已绘制图像
    # 直接在标题里加色块描述即可，色块由 title 颜色体现


# ── 主函数 ──────────────────────────────────────────────────────────────────────

def main(n_samples: int = 4, patch_size: int = 512,
         save_path: str = None, seed: int = SEED,
         files: list = None):
    random.seed(seed)
    np.random.seed(seed)

    img_dir = os.path.join(DATA_ROOT, SPLIT, "all_images")
    gt_dir  = os.path.join(DATA_ROOT, SPLIT, "all_labels")
    txt_dir = os.path.join(DATA_ROOT, SPLIT, "box_label_txt")

    # 指定文件时直接用，否则随机抽取
    if files:
        candidate_files = [f if f.endswith(('.jpg', '.png', '.jpeg')) else f + '.jpg'
                           for f in files]
    else:
        candidate_files = sorted([f for f in os.listdir(img_dir)
                                   if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        random.shuffle(candidate_files)

    # ── 收集有效 patch ─────────────────────────────────────────────────────────
    samples = []
    for fname in candidate_files:
        if not files and len(samples) >= n_samples:
            break
        img_path = os.path.join(img_dir, fname)
        gt_path  = os.path.join(gt_dir,  fname)
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            print(f"  ✗ {fname} 不存在，跳过")
            continue
        result = find_patch_with_strokes(img_path, gt_path, patch_size)
        if result is None:
            print(f"  ✗ {fname} 未找到有效 patch，跳过")
            continue
        Iin_p, Igt_p, cx, cy = result

        # 加载类别 1（手写字迹）掩码，裁剪到 patch 坐标
        txt_name = os.path.splitext(fname)[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_name)
        full_img = cv2.imread(img_path)
        H, W = full_img.shape[:2]
        class1_mask_full = create_class_mask(txt_path, H, W, target_class=1)
        class2_mask_full = create_class_mask(txt_path, H, W, target_class=2)
        class1_mask = class1_mask_full[cy:cy+patch_size, cx:cx+patch_size]
        class2_mask = class2_mask_full[cy:cy+patch_size, cx:cx+patch_size]

        samples.append((fname, Iin_p, Igt_p, class1_mask, class2_mask))
        print(f"  ✓ {fname}  class1覆盖率={class1_mask.mean()*100:.1f}%")

    if not samples:
        print("未找到有效 patch，请检查数据集路径。")
        return

    # ── 对每个样本生成增强（种子已固定，每次结果一致） ──────────────────────────
    augmented = []
    for fname, Iin, Igt, class1_mask, class2_mask in samples:
        stroke_color = _random_vivid_color()

        # ① 仅笔迹换色（class1 区域，class2 显式排除）
        Iin_stroke = recolor_stroke(Iin, Igt, target_color=stroke_color,
                                    class1_mask=class1_mask,
                                    class2_mask=class2_mask,
                                    **STROKE_PARAMS)

        # ② 仅部分印刷文字着色
        Iin_text, Igt_text = colorize_printed_text(
            Iin, Igt,
            **AUG_PARAMS,
            stroke_threshold=STROKE_PARAMS['threshold'],
            stroke_norm_scale=STROKE_PARAMS['norm_scale'],
        )

        # ③ 组合：部分文字着色 + 笔迹换色（class1 限定，class2 排除）
        Iin_both, Igt_both = recolor_stroke_and_tint(
            Iin, Igt,
            stroke_color=stroke_color,
            class1_mask=class1_mask,
            class2_mask=class2_mask,
            threshold=STROKE_PARAMS['threshold'],
            norm_scale=STROKE_PARAMS['norm_scale'],
            **AUG_PARAMS,
        )

        augmented.append({
            'fname':        fname,
            'Iin':          Iin,
            'Igt':          Igt,
            'Iin_stroke':   Iin_stroke,
            'Iin_text':     Iin_text,
            'Igt_text':     Igt_text,
            'Iin_both':     Iin_both,
            'Igt_both':     Igt_both,
            'stroke_color': stroke_color,
        })

    # ── 绘图 ────────────────────────────────────────────────────────────────────
    col_titles = [
        "原图 Iin",
        "原图 GT",
        "笔迹换色\n(Iin)",
        "部分文字着色\n(GT)",
        "部分文字着色\n(Iin)",
        "组合增强\n(Iin)",
        "组合增强\n(GT)",
    ]
    n_cols = len(col_titles)
    n_rows = len(augmented)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.6 * n_rows),
                             dpi=130)
    if n_rows == 1:
        axes = [axes]

    for r, data in enumerate(augmented):
        imgs = [
            data['Iin'],
            data['Igt'],
            data['Iin_stroke'],
            data['Igt_text'],
            data['Iin_text'],
            data['Iin_both'],
            data['Igt_both'],
        ]
        sc = data['stroke_color']
        row_note = f"笔迹色 RGB{sc}"

        for c, img in enumerate(imgs):
            ax = axes[r][c]
            ax.imshow(img)
            ax.axis('off')
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8, fontweight='bold', pad=3)

        # 行标注
        axes[r][0].set_ylabel(
            f"{data['fname']}\n{row_note}",
            fontsize=6.5, rotation=0, labelpad=72, va='center',
        )

    # 分隔线：在第3列左侧画一条竖线（仅文字着色 vs 原始的分界）
    fig.text(0.01, 0.01,
             f"[手写字迹] threshold={STROKE_PARAMS['threshold']}  "
             f"norm_scale={STROKE_PARAMS['norm_scale']}    "
             f"[印刷文字] color_ratio={AUG_PARAMS['color_ratio']}  "
             f"n_colors={AUG_PARAMS['n_colors']}  "
             f"dilation_px={AUG_PARAMS['dilation_px']}  "
             f"min_area={AUG_PARAMS['min_area']}  "
             f"text_threshold={AUG_PARAMS['text_threshold']}",
             fontsize=7, color='#555')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n已保存：{save_path}")
    else:
        plt.savefig("color_aug_preview.png", bbox_inches='tight')
        print("\n已保存：color_aug_preview.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples',  type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--save_path',  type=str, default=None)
    parser.add_argument('--seed',       type=int,   default=SEED)
    parser.add_argument('--files',      type=str,   nargs='+', default=None,
                        help='指定要可视化的文件名，如 268.jpg 87.jpg')
    args = parser.parse_args()
    main(args.n_samples, args.patch_size, args.save_path, args.seed, args.files)
