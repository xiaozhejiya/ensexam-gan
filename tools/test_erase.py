"""
擦除效果推理测试脚本

用法:
    cd ensexam-gan
    python tools/test_erase.py <图片路径> [--ckpt <权重路径>] [--overlap <重叠像素>] [--out <输出目录>]

示例:
    python tools/test_erase.py E:/dataset/SCUT-EnsExam/test/images/001.jpg
    python tools/test_erase.py image.jpg --ckpt ensexam_checkpoints/best.pth
"""
import sys
import os
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config
from networks.generator import Generator

# ── 参数解析 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EnsExam-GAN 擦除推理")
parser.add_argument("image", type=str, help="输入图片路径")
parser.add_argument("--ckpt", type=str, default=None,
                    help="权重文件路径（默认读取 config.yaml 中的 save_dir/latest.pth）")
parser.add_argument("--overlap", type=int, default=32,
                    help="滑动窗口重叠像素数，默认 32")
parser.add_argument("--out", type=str, default=None,
                    help="结果输出目录，默认与输入图片同目录")
args = parser.parse_args()

# ── 路径检查 ──────────────────────────────────────────────────────────────────
test_img_path = Path(args.image)
if not test_img_path.exists():
    print(f"错误: 图片不存在 -> {test_img_path}")
    sys.exit(1)

cfg = load_config()

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


def preprocess(pil_img: Image.Image):
    """RGB 归一化到 [-1, 1]，并 pad 到 PATCH 的整数倍。"""
    img = pil_img.convert("RGB")
    orig_w, orig_h = img.size
    pw = (PATCH - orig_w % PATCH) % PATCH
    ph = (PATCH - orig_h % PATCH) % PATCH
    if pw or ph:
        canvas = Image.new("RGB", (orig_w + pw, orig_h + ph), (255, 255, 255))
        canvas.paste(img, (0, 0))
        img = canvas
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return arr, orig_w, orig_h


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


def ticks(total: int, patch: int, stride: int):
    pts = list(range(0, total - patch + 1, stride))
    if not pts or pts[-1] + patch < total:
        pts.append(total - patch)
    return pts


def mask_to_numpy(t: torch.Tensor) -> np.ndarray:
    """将单通道掩码 tensor [0,1] 转为 uint8 灰度图。"""
    arr = t.squeeze(0).squeeze(0).cpu().numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def infer(arr: np.ndarray):
    h, w, _ = arr.shape
    stride = PATCH - OVERLAP
    result  = np.zeros((h, w, 3), dtype=np.float64)
    ic1_map = np.zeros((h, w, 3), dtype=np.float64)
    ms_map  = np.zeros((h, w),    dtype=np.float64)
    mb_map  = np.zeros((h, w),    dtype=np.float64)
    weight  = np.zeros((h, w),    dtype=np.float64)

    ys = ticks(h, PATCH, stride)
    xs = ticks(w, PATCH, stride)
    total = len(ys) * len(xs)
    done = 0

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch_tensor = to_tensor(arr[y:y + PATCH, x:x + PATCH])
                # Generator 返回: Ms, Mb, Ic4, Ic2, Ic1, Ire, Icomp
                Ms, Mb, _Ic4, _Ic2, Ic1, _Ire, Icomp = G(patch_tensor)
                result[y:y + PATCH, x:x + PATCH]  += to_numpy(Icomp).astype(np.float64)
                ic1_map[y:y + PATCH, x:x + PATCH] += to_numpy(Ic1).astype(np.float64)
                ms_map[y:y + PATCH, x:x + PATCH]  += mask_to_numpy(Ms).astype(np.float64)
                mb_map[y:y + PATCH, x:x + PATCH]  += mask_to_numpy(Mb).astype(np.float64)
                weight[y:y + PATCH, x:x + PATCH] += 1.0
                done += 1
                print(f"    patch {done}/{total}", end="\r")

    print()
    w3    = weight[:, :, np.newaxis]
    icomp = np.clip(result  / w3, 0, 255).astype(np.uint8)
    ic1   = np.clip(ic1_map / w3, 0, 255).astype(np.uint8)
    ms    = np.clip(ms_map  / weight, 0, 255).astype(np.uint8)
    mb    = np.clip(mb_map  / weight, 0, 255).astype(np.uint8)
    return icomp, ic1, ms, mb


# ── 执行推理 ──────────────────────────────────────────────────────────────────
print(f"\n[2] 测试图片: {test_img_path.name}")
pil_orig = Image.open(test_img_path)
print(f"    原图尺寸: {pil_orig.size[0]}x{pil_orig.size[1]}")

arr, orig_w, orig_h = preprocess(pil_orig)
print(f"    Padding 后: {arr.shape[1]}x{arr.shape[0]}  重叠: {OVERLAP}px")

print("\n[3] 开始推理...")
result_arr, ic1_arr, ms_arr, mb_arr = infer(arr)
result_arr = result_arr[:orig_h, :orig_w]
ic1_arr    = ic1_arr[:orig_h, :orig_w]
ms_arr     = ms_arr[:orig_h, :orig_w]
mb_arr     = mb_arr[:orig_h, :orig_w]

# ── 保存结果 ──────────────────────────────────────────────────────────────────
stem = test_img_path.stem
Image.fromarray(result_arr).save(output_dir / f"erased_{stem}.png")
Image.fromarray(ic1_arr).save(output_dir / f"ic1_{stem}.png")
Image.fromarray(ms_arr).save(output_dir / f"ms_{stem}.png")
Image.fromarray(mb_arr).save(output_dir / f"mb_{stem}.png")
print(f"\n[4] 结果已保存至: {output_dir}")
print(f"    erased_{stem}.png  —— 最终擦除结果 Icomp")
print(f"    ic1_{stem}.png     —— CoarseNet 全分辨率输出 Ic1")
print(f"    ms_{stem}.png      —— 软笔画掩码 Ms")
print(f"    mb_{stem}.png      —— 文本块掩码 Mb")
print("    完成！")
