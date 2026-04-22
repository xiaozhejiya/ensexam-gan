"""整页评估工具。"""

import os
from typing import Dict, Optional, Tuple

import cv2
import torch

from utils.eval_metrics import (
    compute_uint8_image_metrics,
    finalize_metric_sums,
    init_metric_sums,
    merge_metric_sums,
)
from utils.page_inference import infer_full_page
from utils.path_utils import normalize_path


def evaluate_full_pages(generator: torch.nn.Module,
                        data_root: str,
                        device: torch.device,
                        phase: str = 'test',
                        overlap: int = 32,
                        save_dir: Optional[str] = None,
                        metric_device: Optional[torch.device] = None) -> Tuple[Dict[str, float], int]:
    """对指定 split 的整页图像进行重建后评估。"""
    root = normalize_path(data_root)
    img_dir = os.path.join(root, phase, 'all_images')
    gt_dir = os.path.join(root, phase, 'all_labels')
    valid_ext = ('.png', '.jpg', '.jpeg')

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f'Image directory not found: {img_dir}')
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f'GT directory not found: {gt_dir}')

    file_names = sorted(name for name in os.listdir(img_dir) if name.endswith(valid_ext))
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    metric_sums = init_metric_sums()
    n_images = 0
    was_training = generator.training
    generator.eval()

    for file_name in file_names:
        gt_path = os.path.join(gt_dir, file_name)
        if not os.path.exists(gt_path):
            continue

        image_path = os.path.join(img_dir, file_name)
        input_bgr = cv2.imread(image_path)
        gt_bgr = cv2.imread(gt_path)
        if input_bgr is None or gt_bgr is None:
            continue

        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        outputs = infer_full_page(generator, input_rgb, device, overlap=overlap)
        pred_rgb = outputs['icomp']

        merge_metric_sums(
            metric_sums,
            compute_uint8_image_metrics(pred_rgb, gt_rgb, device=metric_device),
        )

        if save_dir is not None:
            save_path = os.path.join(save_dir, os.path.splitext(file_name)[0] + '.png')
            cv2.imwrite(save_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

        n_images += 1

    if was_training:
        generator.train()

    return finalize_metric_sums(metric_sums, n_images), n_images
