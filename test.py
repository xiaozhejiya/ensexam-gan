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
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as compute_ms_ssim
from torch.utils.data import DataLoader

from config_loader import load_config
from data.dataset import EnsExamRealDataset
from networks.generator import Generator
from utils.path_utils import normalize_path

# ── 灰度指标用的腐蚀核 ──────────────────────────────────────────────────────
_CROSS_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


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
    sums = {'psnr': 0.0, 'ms_ssim': 0.0, 'mse': 0.0,
            'l1': 0.0, 'age': 0.0, 'peps': 0.0, 'pceps': 0.0}
    n_batches = 0
    n_images = 0
    saved_count = 0

    for Iin, _, _, _, _, _, Igt in test_loader:
        Iin, Igt = Iin.to(device), Igt.to(device)
        *_, Icomp = G(Iin)

        # [-1,1] → [0,1]
        pred = (Icomp.clamp(-1, 1) + 1) / 2
        gt = (Igt.clamp(-1, 1) + 1) / 2

        # ── 批量指标 ──────────────────────────────────────────────
        mse_val = F.mse_loss(pred, gt).item()
        sums['mse'] += mse_val
        sums['l1'] += F.l1_loss(pred, gt).item()
        sums['psnr'] += (10 * math.log10(1.0 / mse_val) if mse_val > 1e-10 else 100.0)
        sums['ms_ssim'] += compute_ms_ssim(pred, gt, data_range=1.0).item()

        # ── 逐图灰度指标：AGE / pEPs / pCEPs ─────────────────────
        pred_np = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        gt_np = (gt.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

        for b in range(pred_np.shape[0]):
            pred_g = cv2.cvtColor(pred_np[b], cv2.COLOR_RGB2GRAY).astype(np.int16)
            gt_g = cv2.cvtColor(gt_np[b], cv2.COLOR_RGB2GRAY).astype(np.int16)
            diff = np.abs(pred_g - gt_g)

            sums['age'] += diff.mean()
            err_mask = (diff > 20).astype(np.uint8)
            sums['peps'] += err_mask.mean()
            sums['pceps'] += cv2.erode(err_mask, _CROSS_KERNEL, iterations=1).mean()

            # 保存擦除结果
            if save_dir is not None:
                out_img = cv2.cvtColor(pred_np[b], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, f'{saved_count:05d}.png'), out_img)
                saved_count += 1

        n_batches += 1
        n_images += pred_np.shape[0]

        print(f"\r  已评估 {n_images} 张图片...", end="", flush=True)

    print()
    return {
        'psnr': sums['psnr'] / n_batches,
        'ms_ssim': sums['ms_ssim'] / n_batches,
        'mse': sums['mse'] / n_batches,
        'l1': sums['l1'] / n_batches,
        'age': sums['age'] / n_images,
        'peps': sums['peps'] / n_images,
        'pceps': sums['pceps'] / n_images,
    }


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
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    data_cfg = cfg['data']
    train_cfg = cfg['train']

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
    print("=" * 60)

    # ── 构建测试集 ────────────────────────────────────────────────────────
    print("\n[1] 构建测试集...")
    data_root = data_cfg['data_root']
    img_size = data_cfg['img_size']
    mask_threshold = data_cfg['mask_threshold']

    test_dataset = EnsExamRealDataset(
        data_root=data_root, img_size=img_size, is_train=False,
        overlap=0, mask_threshold=mask_threshold, aug_cfg=None, phase='test',
    )
    pin = device.type == 'cuda'
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    print(f"    测试集共 {len(test_dataset)} 个 patches")

    # ── 加载模型 ──────────────────────────────────────────────────────────
    print("\n[2] 加载模型权重...")
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

    # ── 评估 ──────────────────────────────────────────────────────────────
    save_dir = None
    if args.save_images:
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[3] 开始评估（擦除结果将保存至 {save_dir}）...")
    else:
        print("\n[3] 开始评估...")

    metrics = evaluate(G, test_loader, device, save_dir=save_dir)

    # ── 输出结果 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f"  PSNR     = {metrics['psnr']:.4f}")
    print(f"  MS-SSIM  = {metrics['ms_ssim']:.4f}")
    print(f"  MSE      = {metrics['mse']:.6f}")
    print(f"  L1       = {metrics['l1']:.6f}")
    print(f"  AGE      = {metrics['age']:.4f}")
    print(f"  pEPs     = {metrics['peps']:.4f}")
    print(f"  pCEPs    = {metrics['pceps']:.4f}")
    print("=" * 60)

    if save_dir:
        print(f"\n擦除结果已保存至: {save_dir}")


if __name__ == '__main__':
    main()
