#!/usr/bin/env python3
"""
tools/build_stroke_library.py

从白底手写 PDF 中提取笔迹 patch，保存为笔迹库供插入增强使用。

提取原理：
    背景为纯白 (255)，diff = clip(255 - rendered_page, 0, 255) 即为笔迹墨量。
    对 diff 灰度图做连通域分析，按面积过滤后逐块裁剪保存。
    由于背景严格为白色，无需 noise_threshold，diff 在背景处天然为 0。

插入时：Iin[dst] -= diff_patch（与 stroke_insert.py 插入公式一致）

用法：
    python tools/build_stroke_library.py 数学.pdf
    python tools/build_stroke_library.py 数学.pdf --out data/stroke_library --dpi 200
    python tools/build_stroke_library.py 语文.pdf --subject 语文 --dpi 300
    python tools/build_stroke_library.py --input-root data/stroke_library --out data/stroke_library

输出目录结构：
    data/stroke_library/
        数学/
            p001_0000.png
            p001_0001.png
            ...
        语文/
            ...
"""

import argparse
from pathlib import Path

import cv2
import fitz          # PyMuPDF
import numpy as np
from PIL import Image


# ── PDF 渲染 ─────────────────────────────────────────────────────────────────

def render_pdf_pages(pdf_path: str, dpi: int = 200) -> list:
    """逐页渲染 PDF，返回 RGB uint8 numpy array 列表。"""
    doc  = fitz.open(pdf_path)
    mat  = fitz.Matrix(dpi / 72, dpi / 72)   # PDF 默认 72 DPI
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3)
        pages.append(img.copy())
    doc.close()
    return pages


# ── 单页笔迹提取 ──────────────────────────────────────────────────────────────

def extract_patches(page_img: np.ndarray,
                    ink_threshold: int  = 200,
                    min_area: int       = 2000,
                    max_area: int       = 500_000,
                    pad: int            = 5,
                    dilate_ksize: int   = 80,
                    tight_bbox: bool    = True) -> list:
    """
    从单页白底手写图像中提取笔迹 diff patch 列表。

    流程：
        1. 灰度化 → 二值化（灰度 < ink_threshold → 墨迹）
        2. 膨胀合并同一计算块内的字符
        3. 连通域分析，按面积过滤
        4. 裁剪 bounding box，计算 diff = clip(255 - crop, 0, 255)

    Args:
        page_img      : RGB uint8，白底手写页面
        ink_threshold : 灰度 < 此值视为墨迹像素（建议 180~220）
        min_area      : 连通域最小面积，过滤噪点（建议 1000~5000）
                        注意：dilate 会将单像素噪点放大到 ~200px，
                        min_area 需要明显大于此值才能有效过滤
        max_area      : 连通域最大面积，过滤整页大块（建议 50000~500000）
        pad           : bounding box 外扩像素
        dilate_ksize  : 膨胀核大小，合并同块内邻近笔画（建议 60~100）
        tight_bbox    : True=用原始墨迹像素计算紧致 bbox（推荐）
                        False=用膨胀后连通域的 bbox（包含大量空白）

    Returns:
        list of RGB uint8 diff patches
    """
    gray       = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY)
    binary_raw = (gray < ink_threshold).astype(np.uint8)   # 原始墨迹，含噪点

    # 开运算（腐蚀→膨胀）：去除孤立噪点，供分块用
    # 注意：不用开运算结果算 tight bbox，因为腐蚀会吃掉细笔画（"-"、"÷"横线等）
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel_open)

    # 膨胀：合并同一计算式内的字符（基于去噪后的 binary）
    if dilate_ksize > 0:
        kernel  = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
        dilated = cv2.dilate(binary_clean, kernel)
    else:
        dilated = binary_clean

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8)

    H, W = page_img.shape[:2]
    patches = []

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        if tight_bbox:
            # 用原始 binary（开运算前）算 tight bbox，保留细笔画的真实边界
            ink_in_label = binary_raw * (labels == i)
            ys, xs = np.where(ink_in_label > 0)
            if len(xs) == 0:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
        else:
            # 直接使用膨胀后连通域的 bbox
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

        x1 = max(0, x0 - pad)
        y1 = max(0, y0 - pad)
        x2 = min(W, x0 + bw + pad)
        y2 = min(H, y0 + bh + pad)

        crop = page_img[y1:y2, x1:x2]

        # diff = clip(255 - crop, 0, 255)
        # 白色背景 (255) → diff = 0；墨迹像素 → diff > 0
        diff = np.clip(255 - crop.astype(np.int16), 0, 255).astype(np.uint8)

        # bbox 存储含 pad 的实际裁剪坐标，debug 可视化与保存范围一致
        patches.append({'diff': diff, 'bbox': (x1, y1, x2 - x1, y2 - y1)})

    return patches


def debug_visualize(page_img: np.ndarray, patches: list, save_path: str):
    """在页面图像上绘制检测到的 bounding box，保存为 PNG 用于调参。"""
    vis = page_img.copy()
    for idx, p in enumerate(patches):
        x0, y0, bw, bh = p['bbox']
        cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (255, 0, 0), 2)
        cv2.putText(vis, str(idx), (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    # 用 PIL 保存，避免 cv2.imwrite 在 Windows 中文路径下静默失败
    Image.fromarray(vis).save(save_path)
    print(f"  [debug] 已保存可视化 → {save_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def build_library(pdf_path: str,
                  out_dir: str      = 'data/stroke_library',
                  subject: str      = None,
                  dpi: int          = 200,
                  ink_threshold: int = 200,
                  min_area: int     = 2000,
                  max_area: int     = 500_000,
                  pad: int          = 20,
                  dilate_ksize: int = 80,
                  tight_bbox: bool  = True,
                  debug: bool       = False) -> int:
    """渲染 PDF → 提取 patch → 按 SCUT-EnsExam 结构保存到笔迹库。

    输出目录结构：
        <out_dir>/<subject>/
            all_images/       渲染的完整页面图（PNG）
            box_label_txt/    bbox 标注（同 SCUT-EnsExam 四边形格式，category=1）
            patches/          提取的 diff patch（供插入增强使用）
            debug_*.png       （仅 --debug 时生成）bounding box 可视化
    """
    pdf_path = Path(pdf_path)
    if subject is None:
        subject = pdf_path.stem

    base_dir   = Path(out_dir) / subject
    img_dir    = base_dir / 'all_images'
    label_dir  = base_dir / 'box_label_txt'
    patch_dir  = base_dir / 'patches'
    for d in (img_dir, label_dir, patch_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[build_stroke_library] PDF   : {pdf_path.name}")
    print(f"[build_stroke_library] 输出  : {base_dir}")
    print(f"[build_stroke_library] 渲染中（{dpi} DPI）...")

    pages = render_pdf_pages(str(pdf_path), dpi=dpi)
    print(f"[build_stroke_library] 共 {len(pages)} 页，开始提取...")

    total = 0
    for page_idx, page_img in enumerate(pages):
        page_name = f"p{page_idx+1:03d}"

        patches = extract_patches(
            page_img,
            ink_threshold = ink_threshold,
            min_area      = min_area,
            max_area      = max_area,
            pad           = pad,
            dilate_ksize  = dilate_ksize,
            tight_bbox    = tight_bbox,
        )

        # 保存完整页面图
        Image.fromarray(page_img).save(str(img_dir / f"{page_name}.png"))

        # 保存 bbox 标注（SCUT-EnsExam 四边形格式）
        # 格式：x0,y0,x1,y1,x2,y2,x3,y3, category
        # 矩形四角：左上→右上→右下→左下，category=1（学生手写）
        label_lines = []
        for p in patches:
            bx, by, bw, bh = p['bbox']   # 含 pad 的裁剪坐标
            x0, y0 = bx,      by
            x1, y1 = bx + bw, by
            x2, y2 = bx + bw, by + bh
            x3, y3 = bx,      by + bh
            label_lines.append(f"{x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3}, 1")
        (label_dir / f"{page_name}.txt").write_text('\n'.join(label_lines), encoding='utf-8')

        if debug:
            debug_visualize(page_img, patches, str(base_dir / f"debug_{page_name}.png"))

        # 保存 diff patch
        for patch_idx, p in enumerate(patches):
            Image.fromarray(p['diff']).save(
                str(patch_dir / f"{page_name}_{patch_idx:04d}.png"))
            total += 1

        print(f"  第 {page_idx+1:2d} 页：{len(patches)} 个 patch")

    print(f"[build_stroke_library] 完成，共保存 {total} 个 patch → {base_dir}")
    return total


def discover_pdfs_by_structure(input_root: str) -> list:
    """按数据集结构发现 PDF，返回 [(pdf_path, subject_name), ...]。

    支持两种常见输入：
        1) 根目录：<root>/<subject>/*.pdf
        2) 科目目录：<subject_dir>/*.pdf
    """
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"输入路径不存在: {root}")

    tasks = []

    # 结构 1：root/subject/*.pdf
    subject_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    for subject_dir in subject_dirs:
        for pdf in sorted(subject_dir.glob('*.pdf')):
            tasks.append((pdf, subject_dir.name))

    # 结构 2：直接传入某个 subject 目录
    if not tasks:
        for pdf in sorted(root.glob('*.pdf')):
            tasks.append((pdf, root.name))

    return tasks


def build_library_from_root(input_root: str,
                            out_dir: str      = 'data/stroke_library',
                            dpi: int          = 200,
                            ink_threshold: int = 200,
                            min_area: int     = 2000,
                            max_area: int     = 500_000,
                            pad: int          = 20,
                            dilate_ksize: int = 80,
                            tight_bbox: bool  = True,
                            debug: bool       = False,
                            split_by_pdf: bool = True) -> int:
    """按给定目录批量提取 PDF。

    默认 split_by_pdf=True 时，输出为：
        <out_dir>/<subject>/<pdf_stem>/{all_images,box_label_txt,patches}
    这样可避免同科目多个 PDF 输出重名覆盖。
    """
    tasks = discover_pdfs_by_structure(input_root)
    if not tasks:
        print(f"[build_stroke_library] 未在 {input_root} 下找到 PDF。")
        return 0

    print(f"[build_stroke_library] 批量模式：共发现 {len(tasks)} 个 PDF")
    total = 0
    for idx, (pdf_path, subject_name) in enumerate(tasks, start=1):
        if split_by_pdf:
            subject = str(Path(subject_name) / pdf_path.stem)
        else:
            subject = subject_name

        print(f"\n[{idx}/{len(tasks)}] 开始处理：{pdf_path}")
        total += build_library(
            pdf_path      = str(pdf_path),
            out_dir       = out_dir,
            subject       = subject,
            dpi           = dpi,
            ink_threshold = ink_threshold,
            min_area      = min_area,
            max_area      = max_area,
            pad           = pad,
            dilate_ksize  = dilate_ksize,
            tight_bbox    = tight_bbox,
            debug         = debug,
        )

    print(f"\n[build_stroke_library] 批量完成，共保存 {total} 个 patch")
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='从白底手写 PDF 提取笔迹 patch，支持单 PDF 与按目录结构批量提取')

    parser.add_argument('pdf', nargs='?', default=None,
        help='手写 PDF 文件路径（白底）；与 --input-root 二选一')
    parser.add_argument('--input-root', default=None,
        help='按结构批量提取的输入路径：支持 <root>/<subject>/*.pdf 或 <subject_dir>/*.pdf')
    parser.add_argument('--out', default='data/stroke_library',
        help='输出根目录（默认 data/stroke_library）')
    parser.add_argument('--subject', default=None,
        help='科目子目录名（默认取 PDF 文件名去扩展名）')
    parser.add_argument('--dpi', type=int, default=200,
        help='渲染 DPI，建议 150~300（默认 200）')
    parser.add_argument('--ink-threshold', type=int, default=200,
        help='灰度 < 此值视为墨迹（默认 200，建议 180~220）')
    parser.add_argument('--min-area', type=int, default=2000,
        help='最小连通域面积，过滤噪点（建议 1000~5000）')
    parser.add_argument('--max-area', type=int, default=500_000,
        help='最大连通域面积，过滤整页大块（默认 500000）')
    parser.add_argument('--pad', type=int, default=20,
        help='bounding box 外扩像素（建议 15~30）')
    parser.add_argument('--dilate', type=int, default=80,
        help='膨胀核大小，合并同块内笔画（建议 60~100）')
    parser.add_argument('--no-tight-bbox', action='store_true',
        help='禁用紧致 bbox，改用膨胀后连通域的宽松 bbox')
    parser.add_argument('--debug', action='store_true',
        help='保存每页的 bounding box 可视化图，用于调参')
    parser.add_argument('--no-split-by-pdf', action='store_true',
        help='批量模式下不按 PDF 名分子目录（可能导致同科目多 PDF 文件名冲突）')

    args = parser.parse_args()

    if args.pdf and args.input_root:
        parser.error('`pdf` 与 `--input-root` 只能二选一。')
    if not args.pdf and not args.input_root:
        parser.error('请提供 `pdf` 或 `--input-root`。')

    if args.input_root:
        build_library_from_root(
            input_root    = args.input_root,
            out_dir       = args.out,
            dpi           = args.dpi,
            ink_threshold = args.ink_threshold,
            min_area      = args.min_area,
            max_area      = args.max_area,
            pad           = args.pad,
            dilate_ksize  = args.dilate,
            tight_bbox    = not args.no_tight_bbox,
            debug         = args.debug,
            split_by_pdf  = not args.no_split_by_pdf,
        )
    else:
        build_library(
            pdf_path      = args.pdf,
            out_dir       = args.out,
            subject       = args.subject,
            dpi           = args.dpi,
            ink_threshold = args.ink_threshold,
            min_area      = args.min_area,
            max_area      = args.max_area,
            pad           = args.pad,
            dilate_ksize  = args.dilate,
            tight_bbox    = not args.no_tight_bbox,
            debug         = args.debug,
        )
