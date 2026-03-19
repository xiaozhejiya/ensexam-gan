import os

import numpy as np
import cv2
from scipy import ndimage


def generate_mask_from_pair(Iin, Igt, threshold=20, debug=False):
    """
    单块（512x512）软笔画掩码生成 - 修正版本
    :param Iin: 单块图像 (H,W,3) RGB 0-255
    :param Igt: 单块GT图 (H,W,3) RGB 0-255
    :param threshold: 差异阈值，默认20
    :param debug: 是否返回中间结果用于调试
    :return: Ms_gt(软掩码), Mb_gt(基础掩码), [debug_dict]
    """
    H, W = Iin.shape[:2]

    # ========== 1. 生成粗笔画掩码 ==========
    # 使用int16避免溢出，计算RGB三通道的平均差异
    diff = np.abs(Iin.astype(np.int16) - Igt.astype(np.int16)).mean(axis=-1)
    print(f"Diff统计: min={diff.min()}, max={diff.max()}, 90%={np.percentile(diff, 90)}")
    coarse_mask = (diff > threshold).astype(np.uint8)

    # ========== 2. 形态学去噪 ==========
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 开运算去除散点噪声 + 闭运算连接断裂笔画
    denoised_mask = cv2.morphologyEx(coarse_mask, cv2.MORPH_OPEN, kernel)
    denoised_mask = cv2.morphologyEx(denoised_mask, cv2.MORPH_CLOSE, kernel)

    # ========== 3. 生成 Mb_gt（基础文本块掩码） ==========
    # 对去噪后的掩码适度膨胀，作为文本区域指导
    Mb_gt = cv2.dilate(denoised_mask, kernel, iterations=2).astype(np.float32)

    # ========== 4. 生成软笔画掩码 Ms_gt ==========
    # 4.1 骨架：笔画收缩1像素（论文明确要求）
    skeleton = cv2.erode(denoised_mask, kernel, iterations=1)

    # 4.2 外边界区域：原始笔画扩张5像素
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    outer_region = cv2.dilate(denoised_mask, kernel_large, iterations=5)

    # 4.3 距离变换：计算outer_region内每个像素到边缘的最短距离
    # 输入要求：uint8, 0=背景, >0=前景；输出：float32，单位像素
    dist_map = cv2.distanceTransform(outer_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 4.4 SAF 参数 (论文明确值)
    alpha = 3.0
    L = 5.0
    exp_neg_alpha = np.exp(-alpha)
    C = (1 + exp_neg_alpha) / (1 - exp_neg_alpha + 1e-8)  # +eps防除零

    # 4.5 向量化计算软掩码（避免循环，提速100x）
    Ms_gt = np.zeros((H, W), dtype=np.float32)

    # 区域划分mask
    mask_skeleton = skeleton > 0  # 骨架区域 → 值=1
    mask_outer = outer_region > 0  # 扩张区域（含骨架）
    mask_middle = mask_outer & (~mask_skeleton)  # 中间环形区域 → 用SAF

    # (a) 骨架区域强制=1
    Ms_gt[mask_skeleton] = 1.0

    # (b) 中间区域应用SAF衰减
    if np.any(mask_middle):
        D = np.clip(dist_map[mask_middle], 0, L)  # 距离截断到[0, L]
        # SAF公式: C * (2/(1+exp(-α*D/L)) - 1)
        exp_term = np.exp(-alpha * D / L)
        saf_vals = C * (2.0 / (1.0 + exp_term + 1e-8) - 1.0)
        Ms_gt[mask_middle] = np.clip(saf_vals, 0.0, 1.0)

    # (c) 外边界及以外区域保持=0（已初始化为0）

    # ========== 5. 调试信息（可选） ==========
    if debug:
        debug_info = {
            'coarse_mask': coarse_mask,
            'denoised_mask': denoised_mask,
            'skeleton': skeleton,
            'outer_region': outer_region,
            'dist_map': dist_map,
            'mask_middle': mask_middle.astype(np.uint8) * 255
        }
        return Ms_gt, Mb_gt, debug_info

    return Ms_gt, Mb_gt

# def generate_mask_from_pair(Iin: np.ndarray, Igt: np.ndarray,
#                              threshold: int = 20,
#                              debug: bool = False):
#     """
#     单块（512×512）软笔画掩码生成。
#
#     Args:
#         Iin: 原始图像，(H, W, 3) RGB uint8
#         Igt: 擦除 GT 图，(H, W, 3) RGB uint8
#         threshold: 判定笔画区域的像素差异阈值，越小掩码越密
#         debug: True 时额外返回中间结果字典
#
#     Returns:
#         Ms_gt: 软笔画掩码，float32 [0, 1]
#         Mb_gt: 文本块掩码，float32 {0, 1}
#         debug_info (仅 debug=True 时返回)
#     """
#     H, W = Iin.shape[:2]
#
#     # 1. 粗笔画掩码：RGB 三通道平均差异 > threshold
#     diff = np.abs(Iin.astype(np.int16) - Igt.astype(np.int16)).mean(axis=-1)
#     coarse_mask = (diff > threshold).astype(np.uint8)
#
#     # 2. 形态学去噪：开运算去散点 + 闭运算连笔画
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     denoised = cv2.morphologyEx(coarse_mask, cv2.MORPH_OPEN, kernel)
#     denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
#
#     # 3. Mb_gt：对去噪掩码膨胀，覆盖文本块区域
#     Mb_gt = cv2.dilate(denoised, kernel, iterations=2).astype(np.float32)
#
#     # 4. Ms_gt：SAF（Stroke Attention Function）软掩码
#     skeleton = cv2.erode(denoised, kernel, iterations=1)
#
#     kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     outer_region = cv2.dilate(denoised, kernel_large, iterations=5)
#
#     dist_map = cv2.distanceTransform(outer_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
#
#     alpha, L = 3.0, 5.0
#     exp_neg_alpha = np.exp(-alpha)
#     C = (1 + exp_neg_alpha) / (1 - exp_neg_alpha + 1e-8)
#
#     Ms_gt = np.zeros((H, W), dtype=np.float32)
#     mask_skeleton = skeleton > 0
#     mask_outer = outer_region > 0
#     mask_middle = mask_outer & (~mask_skeleton)
#
#     Ms_gt[mask_skeleton] = 1.0
#     if np.any(mask_middle):
#         D = np.clip(dist_map[mask_middle], 0, L)
#         saf = C * (2.0 / (1.0 + np.exp(-alpha * D / L) + 1e-8) - 1.0)
#         Ms_gt[mask_middle] = np.clip(saf, 0.0, 1.0)
#
#     if debug:
#         debug_info = {
#             'coarse_mask': coarse_mask,
#             'denoised_mask': denoised,
#             'skeleton': skeleton,
#             'outer_region': outer_region,
#             'dist_map': dist_map,
#             'mask_middle': mask_middle.astype(np.uint8) * 255,
#         }
#         return Ms_gt, Mb_gt, debug_info
#
#     return Ms_gt, Mb_gt


def slide_crop_and_stitch_mask(Iin, Igt, block_size=512, threshold=20,
                               save_path=None, overlap=0):
    """
    滑动裁剪+掩码拼接（支持重叠平滑）
    :param overlap: 重叠像素数，默认0（无重叠），建议设为32-64减少块边界伪影
    """
    H, W = Iin.shape[:2]
    full_Ms = np.zeros((H, W), dtype=np.float32)
    full_Mb = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)  # 用于重叠区域加权平均

    step_h = block_size - overlap
    step_w = block_size - overlap
    num_h = max(1, (H - overlap + step_h - 1) // step_h)
    num_w = max(1, (W - overlap + step_w - 1) // step_w)

    print(f"📐 原图: {W}×{H} | 块: {block_size}×{block_size} | 重叠: {overlap}")
    print(f"🔍 分块: {num_h}×{num_w} = {num_h * num_w} blocks")

    for i in range(num_h):
        for j in range(num_w):
            # 计算裁剪坐标（处理边界）
            y1 = i * step_h
            x1 = j * step_w
            y2 = min(y1 + block_size, H)
            x2 = min(x1 + block_size, W)

            # 反向计算起始点，确保块大小为block_size（边界特殊处理）
            if y2 - y1 < block_size:
                y1 = max(0, y2 - block_size)
            if x2 - x1 < block_size:
                x1 = max(0, x2 - block_size)

            # 裁剪 + 必要时padding
            block_Iin = Iin[y1:y2, x1:x2]
            block_Igt = Igt[y1:y2, x1:x2]
            pad_h = block_size - block_Iin.shape[0]
            pad_w = block_size - block_Iin.shape[1]
            if pad_h > 0 or pad_w > 0:
                block_Iin = cv2.copyMakeBorder(block_Iin, 0, pad_h, 0, pad_w,
                                               cv2.BORDER_REPLICATE)  # 用replicate比constant好
                block_Igt = cv2.copyMakeBorder(block_Igt, 0, pad_h, 0, pad_w,
                                               cv2.BORDER_REPLICATE)

            # 生成掩码
            block_Ms, block_Mb = generate_mask_from_pair(
                block_Iin, block_Igt, threshold)

            # 裁剪padding区域
            block_Ms = block_Ms[:y2 - y1, :x2 - x1]
            block_Mb = block_Mb[:y2 - y1, :x2 - x1]

            # 重叠区域加权融合（避免块边界突变）
            if overlap > 0:
                # 生成该块的权重图（中心高，边缘低）
                h, w = block_Ms.shape
                wy = np.linspace(0.5, 1.0, overlap // 2 + 1)[:-1] if overlap > 0 else [1.0]
                wx = np.linspace(0.5, 1.0, overlap // 2 + 1)[:-1] if overlap > 0 else [1.0]
                weight = np.ones((h, w), dtype=np.float32)
                if overlap > 0:
                    for k in range(overlap // 2):
                        weight[k, :] = np.minimum(weight[k, :], wy[k])
                        weight[-(k + 1), :] = np.minimum(weight[-(k + 1), :], wy[k])
                        weight[:, k] = np.minimum(weight[:, k], wx[k])
                        weight[:, -(k + 1)] = np.minimum(weight[:, -(k + 1)], wx[k])

                # 加权累加
                full_Ms[y1:y2, x1:x2] += block_Ms * weight
                full_Mb[y1:y2, x1:x2] += block_Mb * weight
                weight_map[y1:y2, x1:x2] += weight
            else:
                full_Ms[y1:y2, x1:x2] = block_Ms
                full_Mb[y1:y2, x1:x2] = block_Mb

    # 归一化重叠区域
    if overlap > 0:
        weight_map = np.clip(weight_map, 1e-8, None)
        full_Ms /= weight_map
        full_Mb /= weight_map

    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(f"{save_path}_ms.png", (np.clip(full_Ms, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(f"{save_path}_mb.png", (np.clip(full_Mb, 0, 1) * 255).astype(np.uint8))
        print(f"✅ 掩码已保存: {save_path}_{{ms,mb}}.png")

    return full_Ms, full_Mb


def visualize_mask_generation(Iin, Igt, threshold=20, save_prefix=None):
    """可视化每一步生成过程，排查问题"""
    Ms, Mb, debug = generate_mask_from_pair(Iin, Igt, threshold, debug=True)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    titles = ['Input', 'GT', 'Diff>thresh', 'Denoised',
              'Skeleton', 'Outer Region', 'Distance Map', 'Soft Mask']
    imgs = [Iin, Igt, debug['coarse_mask'] * 255, debug['denoised_mask'] * 255,
            debug['skeleton'] * 255, debug['outer_region'] * 255,
            np.clip(debug['dist_map'] / 5 * 255, 0, 255), np.clip(Ms * 255, 0, 255)]

    for ax, img, title in zip(axes.flat, imgs, titles):
        ax.imshow(img if img.ndim == 3 else img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_debug.png", dpi=150, bbox_inches='tight')
        print(f"📊 调试图已保存: {save_prefix}_debug.png")
    plt.show()

    # 打印统计信息
    print(f"\n📈 掩码统计:")
    print(f"  Ms_gt: min={Ms.min():.3f}, max={Ms.max():.3f}, mean={Ms.mean():.3f}")
    print(f"  Mb_gt: min={Mb.min():.3f}, max={Mb.max():.3f}, mean={Mb.mean():.3f}")
    print(f"  骨架像素: {debug['skeleton'].sum()}, 外区域像素: {debug['outer_region'].sum()}")

if __name__ == "__main__":
    # 读取图像
    Iin = cv2.imread(r"D:\PythonProject1\LLM\adcj\src\adcj\EnsExam\SCUT-EnsExam\SCUT-EnsExam\train\all_images\1.jpg")[:, :, ::-1]  # BGR→RGB
    Igt = cv2.imread(r"D:\PythonProject1\LLM\adcj\src\adcj\EnsExam\SCUT-EnsExam\SCUT-EnsExam\train\all_labels\1.jpg")[:, :, ::-1]

    # 🔍 先单块调试（推荐！）
    # 裁剪中心512x512区域测试
    h, w = Iin.shape[:2]
    cy, cx = h // 2, w // 2
    block_in = Iin[cy - 256:cy + 256, cx - 256:cx + 256]
    block_gt = Igt[cy - 256:cy + 256, cx - 256:cx + 256]

    # 可视化调试每一步
    visualize_mask_generation(block_in, block_gt, threshold=20, save_prefix="./debug")

    # ✅ 确认效果后，全图滑动拼接
    full_Ms, full_Mb = slide_crop_and_stitch_mask(
        Iin, Igt,
        block_size=512,
        threshold=20,
        overlap=32,  # 加32px重叠减少块边界伪影
        save_path="./result/mask_0"
    )