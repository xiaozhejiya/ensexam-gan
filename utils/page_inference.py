"""整页切块推理与融合工具。"""

from typing import Callable, Dict, Optional

import numpy as np
import torch


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 127.5 - 1.0


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _image_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _mask_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def ticks(total: int, patch_size: int, stride: int):
    if total <= patch_size:
        return [0]

    points = list(range(0, total - patch_size + 1, stride))
    if not points or points[-1] + patch_size < total:
        points.append(total - patch_size)
    return points


@torch.no_grad()
def infer_full_page(generator: torch.nn.Module,
                    rgb: np.ndarray,
                    device: torch.device,
                    patch_size: int = 512,
                    overlap: int = 32,
                    progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, np.ndarray]:
    """对整页 RGB 图像执行滑窗推理与 overlap 融合。"""
    arr = _normalize_rgb(rgb)
    height, width, _ = arr.shape
    stride = max(patch_size - overlap, 1)

    result = np.zeros((height, width, 3), dtype=np.float64)
    ic1_map = np.zeros((height, width, 3), dtype=np.float64)
    ms_map = np.zeros((height, width), dtype=np.float64)
    mb_map = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)

    ys = ticks(height, patch_size, stride)
    xs = ticks(width, patch_size, stride)
    total = len(ys) * len(xs)
    done = 0

    for y in ys:
        for x in xs:
            patch_arr = arr[y:y + patch_size, x:x + patch_size]
            patch_h, patch_w = patch_arr.shape[:2]

            if patch_h != patch_size or patch_w != patch_size:
                patch_canvas = np.ones((patch_size, patch_size, 3), dtype=np.float32)
                patch_canvas[:patch_h, :patch_w] = patch_arr
                patch_arr = patch_canvas

            patch_tensor = _to_tensor(patch_arr, device)
            ms, mb, _ic4, _ic2, ic1, _ire, icomp = generator(patch_tensor)
            icomp_patch = _image_to_uint8(icomp)[:patch_h, :patch_w].astype(np.float64)
            ic1_patch = _image_to_uint8(ic1)[:patch_h, :patch_w].astype(np.float64)
            ms_patch = _mask_to_uint8(ms)[:patch_h, :patch_w].astype(np.float64)
            mb_patch = _mask_to_uint8(mb)[:patch_h, :patch_w].astype(np.float64)

            result[y:y + patch_h, x:x + patch_w] += icomp_patch
            ic1_map[y:y + patch_h, x:x + patch_w] += ic1_patch
            ms_map[y:y + patch_h, x:x + patch_w] += ms_patch
            mb_map[y:y + patch_h, x:x + patch_w] += mb_patch
            weight[y:y + patch_h, x:x + patch_w] += 1.0
            done += 1

            if progress_callback is not None:
                progress_callback(done, total)

    weight_rgb = weight[:, :, np.newaxis]
    return {
        'icomp': np.clip(result / weight_rgb, 0, 255).astype(np.uint8),
        'ic1': np.clip(ic1_map / weight_rgb, 0, 255).astype(np.uint8),
        'ms': np.clip(ms_map / weight, 0, 255).astype(np.uint8),
        'mb': np.clip(mb_map / weight, 0, 255).astype(np.uint8),
    }
