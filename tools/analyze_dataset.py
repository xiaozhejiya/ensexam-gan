"""
数据集统计分析：输出每张图的 patch 数分布，为 Reptile 超参数配置提供依据。

用法:
    python tools/analyze_dataset.py
    python tools/analyze_dataset.py --config my.yaml
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config_loader import load_config
from data.dataset import EnsExamRealDataset


def analyze(cfg: dict):
    data_cfg = cfg['data']
    reptile_cfg = cfg.get('reptile', {})
    batch_size = cfg['train']['batch_size']

    print("=" * 55)
    print("数据集统计分析")
    print("=" * 55)

    dataset = EnsExamRealDataset(
        data_root      = data_cfg['data_root'],
        img_size       = data_cfg['img_size'],
        is_train       = True,
        overlap        = data_cfg['overlap'],
        mask_threshold = data_cfg['mask_threshold'],
        aug_cfg        = None,   # 统计时不需要增强
    )

    # 按源图像分组
    groups = defaultdict(list)
    for idx, info in enumerate(dataset.patch_index_map):
        groups[info['img_path']].append(idx)

    patch_counts = sorted([len(v) for v in groups.values()])
    n_images     = len(patch_counts)
    total_patches = sum(patch_counts)

    print(f"\n【数据集基本信息】")
    print(f"  训练图像总数        : {n_images}")
    print(f"  总 patch 数         : {total_patches}")
    print(f"  img_size            : {data_cfg['img_size']}")
    print(f"  overlap             : {data_cfg['overlap']}")

    print(f"\n【每张图的 patch 数分布】")
    print(f"  最小值              : {patch_counts[0]}")
    print(f"  最大值              : {patch_counts[-1]}")
    print(f"  中位数              : {patch_counts[n_images // 2]}")
    print(f"  平均值              : {total_patches / n_images:.1f}")

    # patch 数 < batch_size 的图像（inner loop 跑不满一个 batch）
    too_small = sum(1 for c in patch_counts if c < 2)
    print(f"  patch < 2（会被过滤）: {too_small} 张")

    valid_counts = [c for c in patch_counts if c >= 2]
    n_valid = len(valid_counts)
    print(f"  有效 task 数        : {n_valid}  （patch >= 2）")

    # inner_steps 建议
    median_patches = valid_counts[n_valid // 2]
    steps_1pass    = max(1, median_patches // batch_size)   # 刚好跑完一遍的步数
    recommended_steps = max(1, steps_1pass // 3)            # 约 1/3 遍，防止过拟合

    print(f"\n【Reptile 超参数建议】（基于当前 batch_size={batch_size}）")
    print(f"  中位 task 的 batch 数 : {steps_1pass}  步可跑完一遍数据")
    print(f"  建议 inner_steps      : {recommended_steps}  （约 1/3 遍，防止过拟合单一笔迹）")
    print(f"  建议 n_tasks_per_ep   : {min(4, n_valid)}  （不超过有效 task 总数）")
    print(f"  建议 meta_epochs      : {max(30, n_valid * 2)}  "
          f"（每张图平均被元训练 ~{max(30, n_valid * 2) * min(4, n_valid) / n_valid:.1f} 次）")
    print(f"  inner_lr              : {cfg['train']['lr']}  （与 train.lr 保持一致）")
    print(f"  meta_lr               : 0.1  （Reptile 标准值）")

    # 直方图（ASCII）
    print(f"\n【patch 数直方图（每张图）】")
    bins = [1, 2, 4, 8, 16, 32, 64, 128, 999999]
    labels = ['1', '2-3', '4-7', '8-15', '16-31', '32-63', '64-127', '128+']
    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        cnt = sum(1 for c in patch_counts if lo <= c < hi)
        bar = '█' * cnt
        print(f"  {label:>6} patches : {bar} ({cnt})")

    print("\n" + "=" * 55)
    print("当前 config.yaml 中的 reptile 配置：")
    for k, v in reptile_cfg.items():
        print(f"  {k:<25}: {v}")
    print("=" * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    analyze(load_config(args.config))
