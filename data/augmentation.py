"""
基于 albumentations 的 Paired 数据增强。
albumentations 的 additional_targets 机制保证 Iin 和 Igt 施加完全相同的随机变换。
所有概率和强度参数均从 config.yaml 的 data.augmentation 节读取。
"""
import albumentations as A


def get_train_augmentation(aug_cfg: dict) -> A.Compose:
    """
    构建训练增强流水线。

    Args:
        aug_cfg: config.yaml 中 data.augmentation 子字典

    Returns:
        A.Compose 实例，调用方式：
            result = aug(image=Iin, gt=Igt)
            Iin_aug, Igt_aug = result['image'], result['gt']
    """
    return A.Compose([
        A.HorizontalFlip(
            p=aug_cfg['horizontal_flip_p'],
        ),
        A.VerticalFlip(
            p=aug_cfg['vertical_flip_p'],
        ),
        A.RandomRotate90(
            p=aug_cfg['rotate90_p'],
        ),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg['brightness_limit'],
            contrast_limit=aug_cfg['contrast_limit'],
            p=aug_cfg['brightness_contrast_p'],
        ),
        A.GaussNoise(
            var_limit=tuple(aug_cfg['gauss_noise_var_limit']),  # 噪声方差范围 (min, max)
            p=aug_cfg['gauss_noise_p'],
        ),
    ], additional_targets={'gt': 'image'})  # gt 与 image 施加完全相同的随机变换
