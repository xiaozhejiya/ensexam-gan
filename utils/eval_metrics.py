"""共享评估指标工具。"""

from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as compute_ms_ssim


_CROSS_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
_METRIC_KEYS = ('psnr', 'ms_ssim', 'mse', 'l1', 'age', 'peps', 'pceps')
_PAPER_DISPLAY_SCALES = {
    'psnr': 1.0,
    'ms_ssim': 100.0,
    'mse': 100.0,
    'l1': 1.0,
    'age': 1.0,
    'peps': 100.0,
    'pceps': 100.0,
}
_DISPLAY_SPECS = (
    ('psnr', 'PSNR', 4, None),
    ('ms_ssim', 'MS-SSIM', 4, 2),
    ('mse', 'MSE', 6, 4),
    ('l1', 'L1', 6, None),
    ('age', 'AGE', 4, None),
    ('peps', 'pEPs', 4, 2),
    ('pceps', 'pCEPs', 4, 2),
)


def init_metric_sums() -> Dict[str, float]:
    return {key: 0.0 for key in _METRIC_KEYS}


def merge_metric_sums(total: Dict[str, float], delta: Dict[str, float]) -> None:
    for key in _METRIC_KEYS:
        total[key] += delta[key]


def finalize_metric_sums(metric_sums: Dict[str, float], n_images: int) -> Dict[str, float]:
    if n_images <= 0:
        raise ValueError('n_images must be positive when finalizing metrics.')
    return {key: metric_sums[key] / n_images for key in _METRIC_KEYS}


def to_unit_interval(image: torch.Tensor) -> torch.Tensor:
    return (image.clamp(-1, 1) + 1) / 2


def compute_batch_metric_sums(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """按图片计算一个 batch 的指标和。pred / gt 需在 [0,1]。"""
    mse_vals = F.mse_loss(pred, gt, reduction='none').flatten(1).mean(dim=1)
    l1_vals = F.l1_loss(pred, gt, reduction='none').flatten(1).mean(dim=1)
    psnr_vals = torch.where(
        mse_vals > 1e-10,
        10 * torch.log10(1.0 / mse_vals),
        torch.full_like(mse_vals, 100.0),
    )
    ms_ssim_vals = compute_ms_ssim(pred, gt, data_range=1.0, size_average=False)

    pred_np = (pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    gt_np = (gt.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    metric_sums = {
        'psnr': float(psnr_vals.sum().item()),
        'ms_ssim': float(ms_ssim_vals.sum().item()),
        'mse': float(mse_vals.sum().item()),
        'l1': float(l1_vals.sum().item()),
        'age': 0.0,
        'peps': 0.0,
        'pceps': 0.0,
    }

    for index in range(pred_np.shape[0]):
        pred_gray = cv2.cvtColor(pred_np[index], cv2.COLOR_RGB2GRAY).astype(np.int16)
        gt_gray = cv2.cvtColor(gt_np[index], cv2.COLOR_RGB2GRAY).astype(np.int16)
        diff = np.abs(pred_gray - gt_gray)
        err_mask = (diff > 20).astype(np.uint8)

        metric_sums['age'] += float(diff.mean())
        metric_sums['peps'] += float(err_mask.mean())
        metric_sums['pceps'] += float(cv2.erode(err_mask, _CROSS_KERNEL, iterations=1).mean())

    return metric_sums


def compute_uint8_image_metrics(pred_rgb: np.ndarray,
                                gt_rgb: np.ndarray,
                                device: torch.device = None) -> Dict[str, float]:
    """对单张 uint8 RGB 图像计算指标。"""
    pred_tensor = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
    return finalize_metric_sums(compute_batch_metric_sums(pred_tensor, gt_tensor), 1)


def paper_display_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        key: metrics[key] * _PAPER_DISPLAY_SCALES[key]
        for key in _METRIC_KEYS
    }


def format_metric_block(metrics: Dict[str, float], prefix: str = '  ') -> List[str]:
    paper_metrics = paper_display_metrics(metrics)
    lines = []
    for key, label, raw_precision, paper_precision in _DISPLAY_SPECS:
        raw_text = f"{metrics[key]:.{raw_precision}f}"
        if paper_precision is None:
            lines.append(f"{prefix}{label:<8} = {raw_text}")
            continue

        paper_text = f"{paper_metrics[key]:.{paper_precision}f}"
        lines.append(f"{prefix}{label:<8} = {raw_text}  (paper: {paper_text})")

    return lines
