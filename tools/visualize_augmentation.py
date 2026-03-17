"""
数据增强可视化工具：展示增强前后的 Iin / Igt / Mb（含 box 轮廓叠加），用于验证空间变换一致性。

用法：
    python tools/visualize_augmentation.py                  # 随机抽 4 个 patch
    python tools/visualize_augmentation.py --n 8            # 随机抽 8 个 patch
    python tools/visualize_augmentation.py --index 0 1 2    # 指定 patch 索引
"""
import argparse
import os
import sys
import random

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config
from data.mask_utils import generate_mask_from_pair, generate_mb_from_boxes
from data.augmentation import get_train_augmentation


def overlay_mb_contours(img, mb_mask, color=(0, 255, 0), thickness=2):
    """在图像上叠加 Mb 掩码的轮廓线。"""
    vis = img.copy()
    mb_uint8 = (mb_mask * 255).astype(np.uint8) if mb_mask.max() <= 1 else mb_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mb_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def visualize_sample(info, img_size, mask_threshold, aug, save_path):
    """对单个 patch 生成增强前后的对比图并保存。"""
    # 加载并裁剪
    Iin = cv2.imread(info['img_path'])[:, :, ::-1]
    Igt = cv2.imread(info['gt_path'])[:, :, ::-1]
    Iin = np.ascontiguousarray(Iin[info['y1']:info['y2'], info['x1']:info['x2']])
    Igt = np.ascontiguousarray(Igt[info['y1']:info['y2'], info['x1']:info['x2']])

    # padding
    if info['pad_h'] or info['pad_w']:
        pad_h = img_size - Iin.shape[0]
        pad_w = img_size - Iin.shape[1]
        Iin = cv2.copyMakeBorder(Iin, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        Igt = cv2.copyMakeBorder(Igt, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    # 增强前 Mb
    box_txt = info.get('box_txt_path')
    if box_txt and os.path.exists(box_txt):
        Mb_pre = generate_mb_from_boxes(
            box_txt, info['x1'], info['y1'], info['x2'], info['y2'], img_size,
        )
    else:
        _, Mb_float = generate_mask_from_pair(Iin, Igt, threshold=mask_threshold)
        Mb_pre = (Mb_float > 0.5).astype(np.uint8)

    # 增强前 Ms
    Ms_pre, _ = generate_mask_from_pair(Iin, Igt, threshold=mask_threshold)

    # ── 增强前可视化 ──
    before_iin = overlay_mb_contours(Iin, Mb_pre, color=(0, 255, 0))
    before_igt = overlay_mb_contours(Igt, Mb_pre, color=(0, 255, 0))

    # ── 执行增强 ──
    result = aug(image=Iin, gt=Igt, mb=Mb_pre)
    Iin_aug, Igt_aug, Mb_aug = result['image'], result['gt'], result['mb']

    # 增强后 Ms
    Ms_aug, _ = generate_mask_from_pair(Iin_aug, Igt_aug, threshold=mask_threshold)

    # ── 增强后可视化 ──
    after_iin = overlay_mb_contours(Iin_aug, Mb_aug, color=(0, 255, 0))
    after_igt = overlay_mb_contours(Igt_aug, Mb_aug, color=(0, 255, 0))

    # ── 掩码可视化（灰度→伪彩色）──
    def mask_to_color(mask, cmap=cv2.COLORMAP_JET):
        m = (mask * 255).astype(np.uint8)
        return cv2.applyColorMap(m, cmap)

    Ms_pre_vis  = mask_to_color(Ms_pre)
    Mb_pre_vis  = mask_to_color(Mb_pre.astype(np.float32))
    Ms_aug_vis  = mask_to_color(Ms_aug)
    Mb_aug_vis  = mask_to_color(Mb_aug.astype(np.float32))

    # ── 拼接：4 行 × 2 列（左=增强前 右=增强后）──
    #    行1: Iin + box    行2: Igt + box    行3: Ms    行4: Mb
    h, w = img_size, img_size
    canvas = np.zeros((h * 4, w * 2 + 20, 3), dtype=np.uint8)  # 中间留 20px 分隔
    sep = 20

    def put(row, col, img):
        r, c = row * h, col * (w + sep)
        rgb = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))
        canvas[r:r+h, c:c+w] = rgb

    put(0, 0, before_iin)
    put(0, 1, after_iin)
    put(1, 0, before_igt)
    put(1, 1, after_igt)
    put(2, 0, Ms_pre_vis)
    put(2, 1, Ms_aug_vis)
    put(3, 0, Mb_pre_vis)
    put(3, 1, Mb_aug_vis)

    # 标注文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, label in enumerate(['Iin + Mb box', 'Igt + Mb box', 'Ms (soft stroke)', 'Mb (text block)']):
        y_pos = row * h + 30
        cv2.putText(canvas, f'BEFORE: {label}', (10, y_pos), font, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f'AFTER: {label}', (w + sep + 10, y_pos), font, 0.6, (255, 255, 255), 1)

    # 保存（RGB → BGR）
    cv2.imwrite(save_path, canvas[:, :, ::-1])
    print(f'  saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='数据增强可视化')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--n', type=int, default=4, help='随机抽取的 patch 数量')
    parser.add_argument('--index', type=int, nargs='+', default=None, help='指定 patch 索引')
    parser.add_argument('--output_dir', default='./logs/aug_vis', help='可视化输出目录')
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg['data']
    aug_cfg = data_cfg.get('augmentation')

    if aug_cfg is None:
        print('config.yaml 中未配置 data.augmentation，无法可视化增强。')
        return

    aug = get_train_augmentation(aug_cfg)

    # 复用 dataset 的索引构建逻辑
    from data.dataset import EnsExamRealDataset
    ds = EnsExamRealDataset(
        data_root=data_cfg['data_root'],
        img_size=data_cfg['img_size'],
        is_train=True,
        overlap=data_cfg.get('overlap', 0),
        mask_threshold=data_cfg.get('mask_threshold', 20),
        aug_cfg=None,  # 不在 dataset 内部增强，手动控制
    )

    # 选择 patch
    if args.index is not None:
        indices = args.index
    else:
        indices = random.sample(range(len(ds.patch_index_map)), min(args.n, len(ds.patch_index_map)))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'可视化 {len(indices)} 个 patch，输出到 {args.output_dir}/')

    for i, idx in enumerate(indices):
        info = ds.patch_index_map[idx]
        fname = os.path.splitext(os.path.basename(info['img_path']))[0]
        save_path = os.path.join(args.output_dir, f'aug_vis_{i}_{fname}_y{info["y1"]}x{info["x1"]}.png')
        visualize_sample(info, data_cfg['img_size'], data_cfg.get('mask_threshold', 20), aug, save_path)

    print(f'\n完成！共生成 {len(indices)} 张对比图。')


if __name__ == '__main__':
    main()
