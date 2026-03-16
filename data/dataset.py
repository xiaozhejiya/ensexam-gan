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

from data.mask_utils import generate_mask_from_pair
from data.augmentation import get_train_augmentation


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
                 aug_cfg: dict = None):
        self.data_root = data_root
        self.img_size = img_size
        self.is_train = is_train
        self.overlap = overlap
        self.mask_threshold = mask_threshold
        self.augment = (aug_cfg is not None) and is_train

        if self.augment:
            self.aug = get_train_augmentation(aug_cfg)

        # 归一化到 [-1, 1]（匹配 Generator 输入范围）
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        split = "train" if is_train else "test"
        self.img_dir = os.path.join(data_root, split, "all_images")
        self.gt_dir = os.path.join(data_root, split, "all_labels")

        self.patch_index_map = []
        self._build_patch_index()

    def _build_patch_index(self):
        """扫描数据集，构建 patch 索引表（支持大图滑动裁剪）。"""
        valid_ext = (".png", ".jpg", ".jpeg")
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

                    self.patch_index_map.append({
                        'img_path': img_path,
                        'gt_path': gt_path,
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

        # 3. 数据增强（albumentations additional_targets 保证 Iin/Igt 变换一致）
        if self.augment:
            result = self.aug(image=Iin, gt=Igt)
            Iin, Igt = result['image'], result['gt']

        # 4. 生成软笔画掩码 Ms 和文本块掩码 Mb
        Ms_gt_np, Mb_gt_np = generate_mask_from_pair(Iin, Igt, threshold=self.mask_threshold)

        # 5. 图像归一化到 [-1, 1]
        Iin = self.img_transform(Iin.copy())   # (3, H, W)
        Igt = self.img_transform(Igt.copy())   # (3, H, W)

        # 6. 掩码转 Tensor，形状 (1, H, W)
        Ms_gt = torch.from_numpy(Ms_gt_np).unsqueeze(0).float()
        Mb_gt = torch.from_numpy(Mb_gt_np).unsqueeze(0).float()

        # 7. 多尺度 GT（1/4, 1/2, 1/1），供 CoarseNet 多尺度监督用
        Igt_u = Igt.unsqueeze(0)
        Igt4 = F.interpolate(Igt_u, size=(self.img_size // 4, self.img_size // 4),
                             mode='bilinear', align_corners=False).squeeze(0)
        Igt2 = F.interpolate(Igt_u, size=(self.img_size // 2, self.img_size // 2),
                             mode='bilinear', align_corners=False).squeeze(0)
        Igt1 = Igt

        return Iin, Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt
