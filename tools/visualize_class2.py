"""
将 box_label_txt 中类别 2 的区域从原图裁剪出来，集中展示，
方便确认类别 2 究竟是手写字迹还是批改痕迹。

用法：
    python tools/visualize_class2.py
    python tools/visualize_class2.py --n_boxes 40 --save_path output/class2.png
"""

import argparse
import os
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
from utils.path_utils import normalize_path

DATA_ROOT = normalize_path("~/dataset/SCUT-EnsExam")
SPLIT     = "train"
SEED      = 42


def load_boxes_by_class(txt_path: str, target_class: int) -> list:
    """读取标注文件，返回指定类别的四边形坐标列表。"""
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 9:
                continue
            try:
                cls = int(parts[8])
                if cls != target_class:
                    continue
                coords = [int(p) for p in parts[:8]]
                boxes.append(coords)
            except ValueError:
                continue
    return boxes


def crop_box(img: np.ndarray, coords: list, padding: int = 6) -> np.ndarray:
    """从图像中裁剪四边形的包围盒区域（带 padding）。"""
    pts = np.array(coords).reshape(4, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    H, W = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W, x2 + padding)
    y2 = min(H, y2 + padding)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def main(n_boxes: int = 48, save_path: str = None, seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)

    img_dir = os.path.join(DATA_ROOT, SPLIT, "all_images")
    txt_dir = os.path.join(DATA_ROOT, SPLIT, "box_label_txt")

    # 收集所有类别 2 的 (图像路径, box坐标) 对
    all_crops_info = []
    for fname in sorted(os.listdir(txt_dir)):
        if not fname.endswith('.txt'):
            continue
        txt_path = os.path.join(txt_dir, fname)
        boxes    = load_boxes_by_class(txt_path, target_class=2)
        if not boxes:
            continue
        img_name = fname.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name = fname.replace('.txt', '.png')
            img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        for box in boxes:
            all_crops_info.append((img_path, box))

    print(f"类别 2 的标注框总数：{len(all_crops_info)}")

    # 随机采样
    random.shuffle(all_crops_info)
    selected = all_crops_info[:n_boxes]

    # 裁剪
    crops = []
    cache = {}
    for img_path, box in selected:
        if img_path not in cache:
            cache[img_path] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        crop = crop_box(cache[img_path], box)
        if crop is not None:
            crops.append((os.path.basename(img_path), crop))

    if not crops:
        print("未裁剪到有效区域。")
        return

    # 绘图：每行 8 个
    n_cols = 8
    n_rows = (len(crops) + n_cols - 1) // n_cols
    thumb_h = 100   # 统一缩放高度

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 1.5), dpi=120)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, (fname, crop) in enumerate(crops):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        # 等比缩放到 thumb_h
        h, w = crop.shape[:2]
        tw = max(1, int(w * thumb_h / h))
        thumb = cv2.resize(crop, (tw, thumb_h), interpolation=cv2.INTER_AREA)
        ax.imshow(thumb)
        ax.set_title(fname, fontsize=5, pad=2)
        ax.axis('off')

    # 隐藏多余格子
    for i in range(len(crops), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].axis('off')

    fig.suptitle(f'类别 2 区域采样（共 {len(crops)} 个，seed={seed}）',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = save_path or "output/class2_preview.png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    print(f"已保存：{out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_boxes',   type=int, default=48,
                        help='展示的标注框数量，默认 48')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed',      type=int, default=SEED)
    args = parser.parse_args()
    main(args.n_boxes, args.save_path, args.seed)
