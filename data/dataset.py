"""
数据集加载：将整张图裁剪为 img_size×img_size 的 patch，支持 overlap 滑动和边缘 padding。
"""
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from data.mask_utils import generate_mask_from_pair, generate_mb_from_boxes
from data.augmentation import get_train_augmentation
from tools.color_augment import (
    create_class_mask,
    recolor_stroke,
    colorize_printed_text,
    recolor_stroke_and_tint,
    _random_vivid_color,
)
from tools.stroke_insert import insert_strokes, insert_strokes_from_library


class EnsExamRealDataset(Dataset):
    """
    SCUT-EnsExam 数据集加载器。

    目录结构要求：
        data_root/
        ├── train/
        │   ├── all_images/   ← 带笔记的原始图
        │   └── all_labels/   ← 擦除后的 GT 图（同名文件）
        └── test/
            ├── all_images/
            └── all_labels/

    Args:
        data_root:       数据集根目录
        img_size:        裁剪块边长，模型固定输入尺寸（默认 512）
        is_train:        True 为训练集，False 为测试集
        overlap:         相邻裁剪块的重叠像素，训练时设为 128 可增加样本量
        mask_threshold:  判断笔画区域的像素差异阈值（0~255），越小掩码越密
        augment:         是否启用数据增强（仅 is_train=True 时生效）
    """

    def __init__(self,
                 data_root: str,
                 img_size: int = 512,
                 is_train: bool = True,
                 overlap: int = 0,
                 mask_threshold: int = 20,
                 aug_cfg: dict = None,
                 file_list: list = None,
                 phase: str = "train"):
        """
        Args:
            file_list: 指定使用的图像文件名列表（仅文件名，不含路径）。
                       为 None 时使用对应 split 目录下的全部图像。
                       用于从训练集中划分验证子集时传入不同的文件列表。
        """
        self.data_root = data_root
        self.img_size = img_size
        self.is_train = is_train
        self.overlap = overlap
        self.mask_threshold = mask_threshold
        self.augment = (aug_cfg is not None) and is_train
        self.aug_cfg = aug_cfg or {}
        self.phase = phase

        if self.augment:
            self.aug = get_train_augmentation(aug_cfg)

        # 归一化到 [-1, 1]（匹配 Generator 输入范围）
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        split = "train" if is_train else "test"
        self.img_dir  = os.path.join(data_root, split, "all_images")
        self.gt_dir   = os.path.join(data_root, split, "all_labels")
        self.box_dir  = os.path.join(data_root, split, "box_label_txt")
        self.has_boxes = os.path.isdir(self.box_dir)
        self._file_list = file_list  # None 表示使用全部文件

        self.patch_index_map = []
        self._build_patch_index()

    def _apply_domain_augment(self,
                              Iin: np.ndarray,
                              Igt: np.ndarray,
                              box_txt_path: str) -> tuple:
        """执行字迹/颜色领域增强，返回 (Iin_aug, Igt_aug)。"""
        # 仅在 data.augmentation.domain_augment 配置存在时启用
        if not self.augment:
            return Iin, Igt
        domain_cfg = self.aug_cfg.get('domain_augment', {})
        if not domain_cfg.get('enabled', False):
            return Iin, Igt

        apply_on = set(domain_cfg.get('apply_on', ['train']))
        if self.phase not in apply_on:
            return Iin, Igt

        p_all = float(domain_cfg.get('p', 1.0))
        if np.random.rand() >= p_all:
            return Iin, Igt

        H, W = Iin.shape[:2]
        class1_mask = class2_mask = None
        if box_txt_path and os.path.exists(box_txt_path):
            class1_mask = create_class_mask(box_txt_path, H, W, target_class=1)
            class2_mask = create_class_mask(box_txt_path, H, W, target_class=2)

        # 1) 颜色增强（先改原卷色彩，再做字迹插入，避免新插入字迹被二次改色）
        color_cfg = domain_cfg.get('color_augment', {})
        if color_cfg.get('enabled', False):
            p_color = float(color_cfg.get('p', 0.5))
            if np.random.rand() < p_color:
                mode = color_cfg.get('mode', 'both')
                stroke_params = color_cfg.get('stroke_params', {})
                text_params = color_cfg.get('text_params', {})
                if mode == 'stroke':
                    stroke_color = stroke_params.get('stroke_color', None)
                    if stroke_color == 'random' or stroke_color is None:
                        stroke_color = _random_vivid_color()
                    Iin = recolor_stroke(
                        Iin, Igt,
                        target_color=tuple(stroke_color),
                        threshold=stroke_params.get('threshold', 15),
                        norm_scale=stroke_params.get('norm_scale', 60.0),
                        class1_mask=class1_mask,
                        class2_mask=class2_mask,
                    )
                elif mode == 'text':
                    Iin, Igt = colorize_printed_text(
                        Iin, Igt,
                        color_ratio=text_params.get('color_ratio', 0.25),
                        n_colors=text_params.get('n_colors', 2),
                        dilation_px=text_params.get('dilation_px', 15),
                        min_area=text_params.get('min_area', 300),
                        text_threshold=text_params.get('text_threshold', 180),
                        stroke_threshold=stroke_params.get('threshold', 15),
                        stroke_norm_scale=stroke_params.get('norm_scale', 60.0),
                    )
                else:  # both
                    stroke_color = stroke_params.get('stroke_color', None)
                    if stroke_color == 'random':
                        stroke_color = None
                    Iin, Igt = recolor_stroke_and_tint(
                        Iin, Igt,
                        stroke_color=stroke_color,
                        color_ratio=text_params.get('color_ratio', 0.25),
                        n_colors=text_params.get('n_colors', 2),
                        dilation_px=text_params.get('dilation_px', 15),
                        min_area=text_params.get('min_area', 300),
                        text_threshold=text_params.get('text_threshold', 180),
                        threshold=stroke_params.get('threshold', 15),
                        norm_scale=stroke_params.get('norm_scale', 60.0),
                        class1_mask=class1_mask,
                        class2_mask=class2_mask,
                    )

        # 2) 字迹插入增强（后执行，确保插入字迹颜色保持插入策略本身的分布）
        stroke_cfg = domain_cfg.get('stroke_insert', {})
        if stroke_cfg.get('enabled', False):
            p_stroke = float(stroke_cfg.get('p', 0.5))
            if np.random.rand() < p_stroke:
                mode = stroke_cfg.get('mode', 'library')
                if mode == 'exam':
                    exam_params = stroke_cfg.get('exam_params', {})
                    Iin = insert_strokes(
                        Iin, Igt,
                        class1_mask=class1_mask,
                        class2_mask=class2_mask,
                        n_insert=exam_params.get('n_insert', 5),
                        noise_threshold=exam_params.get('noise_threshold', 30),
                        min_patch_peak=exam_params.get('min_patch_peak', 60),
                        min_area=exam_params.get('min_area', 500),
                        text_threshold=exam_params.get('text_threshold', 210),
                        margin=exam_params.get('margin', 30),
                        return_positions=False,
                    )
                elif mode == 'library':
                    lib_params = stroke_cfg.get('library_params', {})
                    library_dir = lib_params.get('library_dir', None)
                    if library_dir:
                        Iin = insert_strokes_from_library(
                            Iin, Igt,
                            library_dir=library_dir,
                            n_insert=lib_params.get('n_insert', 5),
                            scale_range=tuple(lib_params.get('scale_range', [0.7, 1.3])),
                            angle_range=tuple(lib_params.get('angle_range', [-15, 15])),
                            ink_color=lib_params.get('ink_color', 'random'),
                            text_threshold=lib_params.get('text_threshold', 210),
                            margin=lib_params.get('margin', 30),
                            return_positions=False,
                        )

        return Iin, Igt

    def _build_patch_index(self):
        """扫描数据集，构建 patch 索引表（支持大图滑动裁剪）。"""
        valid_ext = (".png", ".jpg", ".jpeg")
        if self._file_list is not None:
            all_files = [f for f in self._file_list if f.endswith(valid_ext)]
        else:
            all_files = [f for f in os.listdir(self.img_dir) if f.endswith(valid_ext)]

        print(f"正在扫描数据集，构建裁剪索引 (overlap={self.overlap})...")
        for fname in all_files:
            gt_path = os.path.join(self.gt_dir, fname)
            if not os.path.exists(gt_path):
                continue

            img_path = os.path.join(self.img_dir, fname)
            img_temp = cv2.imread(img_path)
            if img_temp is None:
                continue
            H, W = img_temp.shape[:2]
            del img_temp

            step = max(self.img_size - self.overlap, 1)
            num_h = math.ceil((H - self.overlap) / step) if H > self.img_size else 1
            num_w = math.ceil((W - self.overlap) / step) if W > self.img_size else 1

            if H <= self.img_size and W <= self.img_size:
                num_h = num_w = 1

            for i in range(num_h):
                for j in range(num_w):
                    y1 = i * step
                    x1 = j * step
                    y2 = min(y1 + self.img_size, H)
                    x2 = min(x1 + self.img_size, W)

                    if H <= self.img_size:
                        y1, y2 = 0, H
                    if W <= self.img_size:
                        x1, x2 = 0, W

                    # box_label_txt 文件名与图像同名，扩展名改为 .txt
                    box_txt = os.path.join(
                        self.box_dir, os.path.splitext(fname)[0] + '.txt'
                    ) if self.has_boxes else None

                    self.patch_index_map.append({
                        'img_path':     img_path,
                        'gt_path':      gt_path,
                        'box_txt_path': box_txt,
                        'y1': y1, 'y2': y2,
                        'x1': x1, 'x2': x2,
                        'pad_h': (y2 - y1) < self.img_size,
                        'pad_w': (x2 - x1) < self.img_size,
                    })

        print(f"索引构建完成：{len(self.patch_index_map)} 个 patch（来自 {len(all_files)} 张图）")
        assert len(self.patch_index_map) > 0, f"未找到有效样本，请检查目录：{self.img_dir}"

    def __len__(self):
        return len(self.patch_index_map)

    def __getitem__(self, idx):
        info = self.patch_index_map[idx]

        # 1. 加载完整图并裁剪（BGR → RGB）
        Iin = cv2.imread(info['img_path'])[:, :, ::-1]
        Igt = cv2.imread(info['gt_path'])[:, :, ::-1]
        Iin = np.ascontiguousarray(Iin[info['y1']:info['y2'], info['x1']:info['x2']])
        Igt = np.ascontiguousarray(Igt[info['y1']:info['y2'], info['x1']:info['x2']])

        # 2. 边缘 padding（REPLICATE 比黑边伪影少）
        if info['pad_h'] or info['pad_w']:
            pad_h = self.img_size - Iin.shape[0]
            pad_w = self.img_size - Iin.shape[1]
            Iin = cv2.copyMakeBorder(Iin, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            Igt = cv2.copyMakeBorder(Igt, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

        # 3. 生成 Mb（增强前，坐标在原图空间）
        #    有 box_label_txt 时用精确四边形标注；否则退回像素差值+膨胀
        box_txt = info['box_txt_path']
        if box_txt and os.path.exists(box_txt):
            Mb_pre = generate_mb_from_boxes(
                box_txt,
                info['x1'], info['y1'], info['x2'], info['y2'],
                self.img_size,
            )
        else:
            _, Mb_float = generate_mask_from_pair(Iin, Igt, threshold=self.mask_threshold)
            Mb_pre = (Mb_float > 0.5).astype(np.uint8)

        # 4. 领域增强（字迹插入 / 色彩增强）
        Iin, Igt = self._apply_domain_augment(Iin, Igt, box_txt)
        # 新增字迹后，用差值掩码补充 Mb，避免监督遗漏新增区域
        _, Mb_from_aug = generate_mask_from_pair(Iin, Igt, threshold=self.mask_threshold)
        Mb_pre = np.maximum(Mb_pre, (Mb_from_aug > 0.5).astype(np.uint8))

        # 5. albumentations 数据增强（Iin/Igt/Mb 施加相同的空间变换）
        if self.augment:
            result = self.aug(image=Iin, gt=Igt, mb=Mb_pre)
            Iin, Igt, Mb_pre = result['image'], result['gt'], result['mb']

        # 6. 增强后从像素差值生成 Ms；Mb 直接用增强后的结果
        Ms_gt_np, _ = generate_mask_from_pair(Iin, Igt, threshold=self.mask_threshold)
        Mb_gt_np = Mb_pre.astype(np.float32)

        # 7. 图像归一化到 [-1, 1]
        Iin = self.img_transform(Iin.copy())   # (3, H, W)
        Igt = self.img_transform(Igt.copy())   # (3, H, W)

        # 8. 掩码转 Tensor，形状 (1, H, W)
        Ms_gt = torch.from_numpy(Ms_gt_np).unsqueeze(0).float()
        Mb_gt = torch.from_numpy(Mb_gt_np).unsqueeze(0).float()

        # 9. 多尺度 GT（1/4, 1/2, 1/1），供 CoarseNet 多尺度监督用
        Igt_u = Igt.unsqueeze(0)
        Igt4 = F.interpolate(Igt_u, size=(self.img_size // 4, self.img_size // 4),
                             mode='bilinear', align_corners=False).squeeze(0)
        Igt2 = F.interpolate(Igt_u, size=(self.img_size // 2, self.img_size // 2),
                             mode='bilinear', align_corners=False).squeeze(0)
        Igt1 = Igt

        return Iin, Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt
