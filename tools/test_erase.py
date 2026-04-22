"""
擦除效果推理测试脚本

用法:
    cd ensexam-gan
    python tools/test_erase.py <图片路径> [--ckpt <权重路径>] [--overlap <重叠像素>] [--out <输出目录>]
    python tools/test_erase.py <图片路径> --disable-auto-scale

示例:
    python tools/test_erase.py E:/dataset/SCUT-EnsExam/test/images/001.jpg
    python tools/test_erase.py image.jpg --ckpt ensexam_checkpoints/best.pth
"""
import sys
import os
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config
from networks.generator import Generator
from utils.page_inference import infer_full_page

# ── 参数解析 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EnsExam-GAN 擦除推理")
parser.add_argument("image", type=str, help="输入图片路径")
parser.add_argument("--ckpt", type=str, default=None,
                    help="权重文件路径（默认读取 config.yaml 中的 save_dir/latest.pth）")
parser.add_argument("--overlap", type=int, default=32,
                    help="滑动窗口重叠像素数，默认 32")
parser.add_argument("--out", type=str, default=None,
                    help="结果输出目录，默认与输入图片同目录")
parser.add_argument("--disable-auto-scale", action="store_true",
                    help="关闭自动尺度归一化，方便与原始流程做 A/B 对比")
args = parser.parse_args()

# ── 路径检查 ──────────────────────────────────────────────────────────────────
test_img_path = Path(args.image)
if not test_img_path.exists():
    print(f"错误: 图片不存在 -> {test_img_path}")
    sys.exit(1)

cfg = load_config()
inference_cfg = cfg.get("inference", {})
auto_scale_cfg = inference_cfg.get("auto_scale", {})

if args.ckpt:
    ckpt_path = Path(args.ckpt)
else:
    ckpt_path = Path(cfg["train"]["save_dir"]) / "latest.pth"

if not ckpt_path.exists():
    print(f"错误: 找不到权重文件 -> {ckpt_path}")
    print("请先训练模型，或用 --ckpt 指定权重路径")
    sys.exit(1)

output_dir = Path(args.out) if args.out else Path(__file__).parent.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# ── 环境信息 ──────────────────────────────────────────────────────────────────
print("=" * 50)
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA 可用 : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 50)

# ── 加载模型 ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[1] 加载权重: {ckpt_path}")

G = Generator(cfg["model"]).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
G.load_state_dict(ckpt["G_state_dict"])
G.eval()
print(f"    权重加载成功 (epoch={ckpt.get('epoch', '?')})，运行设备: {device}")

# ── 推理工具函数 ──────────────────────────────────────────────────────────────
PATCH = 512
OVERLAP = args.overlap
AUTO_SCALE_ENABLED = bool(auto_scale_cfg.get("enabled", False)) and not args.disable_auto_scale


def _ensure_odd(value: int) -> int:
    value = max(int(value), 3)
    return value if value % 2 == 1 else value + 1


def estimate_text_height(rgb: np.ndarray, scale_cfg: dict) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        _ensure_odd(scale_cfg.get("adaptive_block_size", 31)),
        float(scale_cfg.get("adaptive_c", 15)),
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = []
    for idx in range(1, num_labels):
        _, _, w, h, area = stats[idx]
        if h < 4 or h > 120:
            continue
        if w < 2 or w > 160:
            continue
        if area < 8 or area > 4000:
            continue

        fill_ratio = area / float(max(w * h, 1))
        aspect_ratio = w / float(max(h, 1))
        if fill_ratio < 0.08 or fill_ratio > 0.95:
            continue
        if aspect_ratio < 0.08 or aspect_ratio > 12.0:
            continue

        heights.append(int(h))

    min_components = int(scale_cfg.get("min_components", 30))
    if len(heights) < min_components:
        return {
            "ok": False,
            "reason": "insufficient_components",
            "component_count": len(heights),
        }

    lo = float(np.percentile(heights, 10))
    hi = float(np.percentile(heights, 90))
    filtered = [h for h in heights if lo <= h <= hi]
    if len(filtered) < min_components:
        return {
            "ok": False,
            "reason": "unstable_distribution",
            "component_count": len(filtered),
        }

    return {
        "ok": True,
        "reason": "ok",
        "component_count": len(filtered),
        "median_height": float(np.median(filtered)),
        "p10": lo,
        "p90": hi,
    }


def compute_auto_scale(rgb: np.ndarray, scale_cfg: dict) -> dict:
    reference_height = float(scale_cfg.get("reference_text_height", 16.0))
    trigger_ratio = float(scale_cfg.get("trigger_ratio", 1.15))
    min_scale = float(scale_cfg.get("min_scale", 0.5))
    max_scale = float(scale_cfg.get("max_scale", 1.0))
    upper_trigger = reference_height * trigger_ratio
    lower_trigger = reference_height / max(trigger_ratio, 1e-6)

    scale_info = {
        "enabled": True,
        "applied": False,
        "scale": 1.0,
        "reference_height": reference_height,
        "trigger_ratio": trigger_ratio,
        "upper_trigger": upper_trigger,
        "lower_trigger": lower_trigger,
        "reason": "within_range",
    }

    text_info = estimate_text_height(rgb, scale_cfg)
    scale_info.update(text_info)
    if not text_info.get("ok", False):
        scale_info["reason"] = text_info.get("reason", "estimate_failed")
        return scale_info

    estimated_height = float(text_info["median_height"])
    raw_scale = reference_height / max(estimated_height, 1e-6)
    clipped_scale = min(max(raw_scale, min_scale), max_scale)
    scale_info.update({
        "estimated_height": estimated_height,
        "raw_scale": raw_scale,
        "clipped_scale": clipped_scale,
    })

    if lower_trigger <= estimated_height <= upper_trigger:
        scale_info["reason"] = "within_range"
        return scale_info

    if estimated_height > upper_trigger:
        target_reason = "scaled_down"
    elif estimated_height < lower_trigger:
        if max_scale <= 1.0:
            scale_info["reason"] = "upscale_disabled"
            return scale_info
        target_reason = "scaled_up"
    else:
        scale_info["reason"] = "within_range"
        return scale_info

    if 0.999 <= clipped_scale <= 1.001:
        scale_info["reason"] = "clipped_to_noop"
        return scale_info

    scale_info["applied"] = True
    scale_info["scale"] = clipped_scale
    scale_info["reason"] = target_reason
    return scale_info


def maybe_apply_auto_scale(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    scale_info = {
        "enabled": AUTO_SCALE_ENABLED,
        "applied": False,
        "scale": 1.0,
        "reference_height": float(auto_scale_cfg.get("reference_text_height", 16.0)),
        "trigger_ratio": float(auto_scale_cfg.get("trigger_ratio", 1.15)),
        "reason": "disabled_by_flag" if args.disable_auto_scale else "disabled_in_config",
    }

    if not AUTO_SCALE_ENABLED:
        return rgb, scale_info

    scale_info = compute_auto_scale(rgb, auto_scale_cfg)
    if not scale_info.get("applied", False):
        return rgb, scale_info

    scale = float(scale_info["scale"])
    resized_w = max(1, int(round(rgb.shape[1] * scale)))
    resized_h = max(1, int(round(rgb.shape[0] * scale)))
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    scale_info["resized_width"] = resized_w
    scale_info["resized_height"] = resized_h
    return resized, scale_info


# ── 执行推理 ──────────────────────────────────────────────────────────────────
print(f"\n[2] 测试图片: {test_img_path.name}")
pil_orig = Image.open(test_img_path)
print(f"    原图尺寸: {pil_orig.size[0]}x{pil_orig.size[1]}")
orig_w, orig_h = pil_orig.size

print("\n[3] 自动尺度归一化...")
inference_rgb, scale_info = maybe_apply_auto_scale(pil_orig)
if not scale_info.get("enabled", False):
    reason = "命令行已关闭" if args.disable_auto_scale else "config 中已关闭"
    print(f"    状态: {reason}")
    print(f"    尺度处理后尺寸: {orig_w}x{orig_h}")
elif not scale_info.get("ok", True):
    print(
        f"    状态: 回退到原始尺度"
        f"（原因: {scale_info.get('reason')}，候选连通域: {scale_info.get('component_count', 0)}）"
    )
    print(f"    尺度处理后尺寸: {orig_w}x{orig_h}")
else:
    print(
        f"    参考高度: {scale_info['reference_height']:.1f}px | "
        f"当前页: {scale_info['estimated_height']:.1f}px | "
        f"候选连通域: {scale_info['component_count']}"
    )
    print(
        f"    触发区间: < {scale_info['lower_trigger']:.1f}px 放大 | "
        f"> {scale_info['upper_trigger']:.1f}px 缩小"
    )
    if scale_info.get("applied", False):
        scaled_w = scale_info["resized_width"]
        scaled_h = scale_info["resized_height"]
        print(
            f"    已触发自动{'放大' if scale_info['reason'] == 'scaled_up' else '缩小'}: "
            f"scale={scale_info['scale']:.3f} -> "
            f"{scaled_w}x{scaled_h}"
        )
        print(
            f"    像素变化: {orig_w * orig_h} -> {scaled_w * scaled_h} "
            f"({scaled_w * scaled_h / max(orig_w * orig_h, 1):.3f}x)"
        )
    else:
        if scale_info.get("reason") == "upscale_disabled":
            print("    状态: 未缩放（原因: 当前页偏小，但配置禁止放大）")
        else:
            print(f"    状态: 未缩放（原因: {scale_info.get('reason')}）")
    print(f"    尺度处理后尺寸: {inference_rgb.shape[1]}x{inference_rgb.shape[0]}")

proc_h, proc_w = inference_rgb.shape[:2]
print(f"    推理尺寸: {proc_w}x{proc_h}")
print(f"    全图不做补边，边缘 patch 按需补齐到 {PATCH}x{PATCH}  重叠: {OVERLAP}px")

print("\n[4] 开始推理...")
outputs = infer_full_page(
    G,
    inference_rgb,
    device,
    patch_size=PATCH,
    overlap=OVERLAP,
    progress_callback=lambda done, total: print(f"    patch {done}/{total}", end="\r"),
)
print()
result_arr = outputs['icomp'][:proc_h, :proc_w]
ic1_arr = outputs['ic1'][:proc_h, :proc_w]
ms_arr = outputs['ms'][:proc_h, :proc_w]
mb_arr = outputs['mb'][:proc_h, :proc_w]

if scale_info.get("applied", False):
    result_arr = cv2.resize(result_arr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    ic1_arr = cv2.resize(ic1_arr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    ms_arr = cv2.resize(ms_arr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mb_arr = cv2.resize(mb_arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    print(f"    输出恢复尺寸: {orig_w}x{orig_h}")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
stem = test_img_path.stem
Image.fromarray(result_arr).save(output_dir / f"erased_{stem}.png")
Image.fromarray(ic1_arr).save(output_dir / f"ic1_{stem}.png")
Image.fromarray(ms_arr).save(output_dir / f"ms_{stem}.png")
Image.fromarray(mb_arr).save(output_dir / f"mb_{stem}.png")
print(f"\n[5] 结果已保存至: {output_dir}")
print(f"    erased_{stem}.png  —— 最终擦除结果 Icomp")
print(f"    ic1_{stem}.png     —— CoarseNet 全分辨率输出 Ic1")
print(f"    ms_{stem}.png      —— 软笔画掩码 Ms")
print(f"    mb_{stem}.png      —— 文本块掩码 Mb")
print("    完成！")
