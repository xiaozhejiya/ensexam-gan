"""
测试脚本：在测试集上评估指定权重的表现。

用法:
    python test.py --weights <权重路径>
    python test.py --weights checkpoints/ensexam/best.pth
    python test.py --weights checkpoints/ensexam/best.pth --config my_cfg.yaml
    python test.py --weights checkpoints/ensexam/best.pth --batch-size 8 --save-images

指标:
    PSNR, MS-SSIM, MSE, L1, AGE, pEPs, pCEPs
"""
import argparse
import os
import sys

import cv2
import torch
from torch.utils.data import DataLoader

from config_loader import load_config
from data.dataset import EnsExamRealDataset
from networks.generator import Generator
from utils.eval_metrics import (
    compute_batch_metric_sums,
    finalize_metric_sums,
    format_metric_block,
    init_metric_sums,
    merge_metric_sums,
    to_unit_interval,
)
from utils.page_eval import evaluate_full_pages
from utils.path_utils import normalize_path


@torch.no_grad()
def evaluate(G: Generator, test_loader: DataLoader, device: torch.device,
             save_dir: str = None) -> dict:
    """
    在测试集上计算全量图像质量指标：
      PSNR、MS-SSIM、MSE、L1、AGE、pEPs、pCEPs

    Args:
        G:           生成器模型（已加载权重）
        test_loader: 测试集 DataLoader
        device:      运行设备
        save_dir:    若不为 None，则将擦除结果保存至该目录

    Returns:
        包含各指标平均值的字典
    """
    G.eval()
    sums = init_metric_sums()
    n_images = 0
    saved_count = 0

    for Iin, _, _, _, _, _, Igt in test_loader:
        Iin, Igt = Iin.to(device), Igt.to(device)
        *_, Icomp = G(Iin)

        # [-1,1] → [0,1]
        pred = to_unit_interval(Icomp)
        gt = to_unit_interval(Igt)

        merge_metric_sums(sums, compute_batch_metric_sums(pred, gt))

        pred_np = (pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype('uint8')

        for b in range(pred_np.shape[0]):
            # 保存擦除结果
            if save_dir is not None:
                out_img = cv2.cvtColor(pred_np[b], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, f'{saved_count:05d}.png'), out_img)
                saved_count += 1

        n_images += pred_np.shape[0]

        print(f"\r  已评估 {n_images} 张图片...", end="", flush=True)

    print()
    return finalize_metric_sums(sums, n_images)


def main():
    parser = argparse.ArgumentParser(
        description="EnsExam-GAN 测试脚本：在测试集上评估指定权重的表现")
    parser.add_argument('--weights', type=str, required=True,
                        help='权重文件路径（.pth），例如 checkpoints/ensexam/best.pth')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径（默认 config.yaml）')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='测试 batch size（默认使用 config 中的 batch_size）')
    parser.add_argument('--device', type=str, default='auto',
                        help='运行设备，auto / cpu / cuda:0（默认 auto）')
    parser.add_argument('--save-images', action='store_true',
                        help='是否保存擦除结果图片')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='保存擦除结果的目录（默认 ./test_results）')
    parser.add_argument('--eval-mode', type=str, choices=['patch', 'page', 'both'], default=None,
                        help='评估模式：patch / page / both（默认读取 config.evaluation.standalone_test_mode）')
    parser.add_argument('--page-overlap', type=int, default=None,
                        help='整页评估时的滑窗重叠像素（默认读取 config.evaluation.page_overlap）')
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    eval_cfg = cfg.get('evaluation', {})

    # ── 检查权重文件 ──────────────────────────────────────────────────────
    weights_path = normalize_path(args.weights)
    if not os.path.exists(weights_path):
        print(f"错误: 找不到权重文件 -> {weights_path}")
        sys.exit(1)

    # ── 设备 ──────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    batch_size = args.batch_size if args.batch_size is not None else train_cfg['batch_size']
    num_workers = train_cfg['num_workers']
    if num_workers == 0 and os.name != 'nt':
        num_workers = min(4, os.cpu_count() or 1)
    eval_mode = args.eval_mode if args.eval_mode is not None else eval_cfg.get('standalone_test_mode', 'both')
    page_overlap = args.page_overlap if args.page_overlap is not None else eval_cfg.get('page_overlap', 32)
    if eval_mode not in {'patch', 'page', 'both'}:
        raise ValueError(f'未知评估模式: {eval_mode}')

    # ── 环境信息 ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("EnsExam-GAN 测试集评估")
    print("=" * 60)
    print(f"PyTorch     : {torch.__version__}")
    print(f"设备        : {device}")
    if device.type == 'cuda':
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"权重文件    : {weights_path}")
    print(f"配置文件    : {args.config}")
    print(f"Batch Size  : {batch_size}")
    print(f"评估模式    : {eval_mode}")
    if eval_mode in ('page', 'both'):
        print(f"Page Overlap: {page_overlap}")
    print("=" * 60)

    data_root = data_cfg['data_root']
    img_size = data_cfg['img_size']
    mask_threshold = data_cfg['mask_threshold']
    pin = device.type == 'cuda'
    test_loader = None
    patch_metrics = None
    page_metrics = None
    page_count = 0
    patch_save_dir = None
    page_save_dir = None
    step = 1

    if eval_mode in ('patch', 'both'):
        print(f"\n[{step}] 构建 patch 测试集...")
        test_dataset = EnsExamRealDataset(
            data_root=data_root, img_size=img_size, is_train=False,
            overlap=0, mask_threshold=mask_threshold, aug_cfg=None, phase='test',
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False, pin_memory=pin,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )
        print(f"    测试集共 {len(test_dataset)} 个 patches")
        step += 1

    # ── 加载模型 ──────────────────────────────────────────────────────────
    print(f"\n[{step}] 加载模型权重...")
    G = Generator(cfg=cfg['model']).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # 兼容不同的 checkpoint 格式
    if 'G_state_dict' in ckpt:
        G.load_state_dict(ckpt['G_state_dict'])
        epoch_info = ckpt.get('epoch', '?')
        print(f"    权重加载成功（epoch={epoch_info}）")
    elif 'state_dict' in ckpt:
        G.load_state_dict(ckpt['state_dict'])
        print("    权重加载成功（state_dict 格式）")
    else:
        # 尝试直接作为 state_dict 加载
        G.load_state_dict(ckpt)
        print("    权重加载成功（裸 state_dict 格式）")
    step += 1

    # ── 评估 ──────────────────────────────────────────────────────────────
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n[{step}] 开始评估（结果将保存至 {args.output_dir}）...")
    else:
        print(f"\n[{step}] 开始评估...")

    if eval_mode in ('patch', 'both'):
        patch_save_dir = args.output_dir if eval_mode == 'patch' and args.save_images else None
        if eval_mode == 'both' and args.save_images:
            patch_save_dir = os.path.join(args.output_dir, 'patches')
            os.makedirs(patch_save_dir, exist_ok=True)
        print("    [patch] 开始 patch 级评估...")
        patch_metrics = evaluate(G, test_loader, device, save_dir=patch_save_dir)

    if eval_mode in ('page', 'both'):
        page_save_dir = args.output_dir if eval_mode == 'page' and args.save_images else None
        if eval_mode == 'both' and args.save_images:
            page_save_dir = os.path.join(args.output_dir, 'pages')
            os.makedirs(page_save_dir, exist_ok=True)
        print(f"    [page] 开始整页评估（overlap={page_overlap}px，tile_batch={batch_size}）...")
        page_metrics, page_count = evaluate_full_pages(
            G,
            data_root=data_root,
            device=device,
            phase='test',
            overlap=page_overlap,
            save_dir=page_save_dir,
            metric_device=device,
            infer_batch_size=batch_size,
            verbose=True,
        )

    # ── 输出结果 ──────────────────────────────────────────────────────────
    if patch_metrics is not None:
        print("\n" + "=" * 60)
        print("Patch 级测试集评估结果")
        print("=" * 60)
        for line in format_metric_block(patch_metrics):
            print(line)
        print("=" * 60)

    if page_metrics is not None:
        print("\n" + "=" * 60)
        print(f"Page 级测试集评估结果（{page_count} 张整页）")
        print("=" * 60)
        for line in format_metric_block(page_metrics):
            print(line)
        print("=" * 60)

    if args.save_images:
        print("\n结果已保存至：")
        if patch_save_dir is not None:
            print(f"  Patch 输出: {patch_save_dir}")
        if page_save_dir is not None:
            print(f"  Page 输出 : {page_save_dir}")


if __name__ == '__main__':
    main()
