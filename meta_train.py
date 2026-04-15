"""
Reptile 元训练入口：学习对所有试卷笔迹风格都泛化的初始参数。

用法:
    python meta_train.py                     # 使用默认 config.yaml
    python meta_train.py --config my.yaml

完成后在 config.yaml 中设置：
    resume: true
    resume_path: ./reptile_checkpoints/reptile_meta_init.pth
再运行 python train.py 进行二次训练。
"""
import argparse
import csv
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config, save_config
from data.dataset import EnsExamRealDataset
from losses.losses import EnsExamLoss
from networks.discriminator import Discriminator
from networks.generator import Generator
from tools.reptile import ReptileMetaLearner
from train import setup_device, wrap_model, unwrap_model, set_seed


def setup_logger(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, 'meta_train.log')

    logger = logging.getLogger('meta_train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s', datefmt='%H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt); logger.addHandler(fh)

    logger.info(f"日志文件：{log_path}")
    return logger


def build_task_loaders(dataset: EnsExamRealDataset,
                       batch_size: int,
                       pin_memory: bool,
                       min_patches: int = 2) -> list:
    """按源图像分组，每张图 → 一个 task DataLoader（每个 task = 一个学生的笔迹）。"""
    groups = defaultdict(list)
    for idx, info in enumerate(dataset.patch_index_map):
        groups[info['img_path']].append(idx)

    loaders = []
    for indices in groups.values():
        if len(indices) < min_patches:
            continue
        subset = Subset(dataset, indices)
        loader = DataLoader(subset,
                            batch_size=min(batch_size, len(indices)),
                            shuffle=True, num_workers=0,
                            drop_last=False, pin_memory=pin_memory)
        loaders.append(loader)

    return loaders


def meta_train(cfg: dict):
    train_cfg   = cfg['train']
    data_cfg    = cfg['data']
    reptile_cfg = cfg['reptile']

    # 创建本次运行目录
    base_dir  = reptile_cfg.get('save_dir', './checkpoints')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = os.path.join(base_dir, 'meta', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, 'config.yaml'))

    logger = setup_logger(run_dir)

    seed = train_cfg.get('seed', None)
    reproducibility_mode = train_cfg.get('reproducibility_mode', 'statistical')
    if seed is not None:
        set_seed(seed, mode=reproducibility_mode)
        logger.info(f"随机种子已固定：{seed}（mode={reproducibility_mode}）")

    device, gpu_ids = setup_device(train_cfg)
    if len(gpu_ids) > 1:
        logger.info(f"多卡训练：GPU {gpu_ids}，主设备 {device}")
    else:
        logger.info(f"使用设备：{device}")

    # 数据集
    dataset = EnsExamRealDataset(
        data_root      = data_cfg['data_root'],
        img_size       = data_cfg['img_size'],
        is_train       = True,
        overlap        = data_cfg['overlap'],
        mask_threshold = data_cfg['mask_threshold'],
        aug_cfg        = data_cfg.get('augmentation'),
        phase          = 'meta',
    )

    task_loaders   = build_task_loaders(dataset, train_cfg['batch_size'],
                                        pin_memory=device.type == 'cuda')
    n_tasks_per_ep = reptile_cfg['n_tasks_per_episode']
    logger.info(f"共 {len(task_loaders)} 个 task | 每 episode 采样 {n_tasks_per_ep} 个")

    if len(task_loaders) < n_tasks_per_ep:
        raise ValueError(
            f"可用 task 数 ({len(task_loaders)}) < n_tasks_per_episode ({n_tasks_per_ep})"
        )

    # 模型（Reptile 的 inner loop 需要操作 state_dict，wrap 后传入 ReptileMetaLearner）
    G         = Generator(cfg=cfg['model']).to(device)
    D         = Discriminator().to(device)
    criterion = EnsExamLoss(cfg=cfg['loss']).to(device)
    G         = wrap_model(G, gpu_ids)
    D         = wrap_model(D, gpu_ids)

    meta_learner = ReptileMetaLearner(G, D, criterion, device, cfg)

    meta_epochs = reptile_cfg['meta_epochs']
    save_every  = reptile_cfg.get('save_every_n_epochs', 10)
    log_every   = reptile_cfg.get('log_every_n_epochs', 10)

    # CSV 日志
    csv_path = os.path.join(run_dir, 'meta_loss_history.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['episode', 'loss_G', 'loss_D'])

    # 滑动窗口：用于计算最近 log_every 个 episode 的平均损失
    window_G = window_D = 0.0
    t_start  = time.time()

    # 元训练循环
    for episode in tqdm(range(meta_epochs), desc='Meta-train', ncols=100):
        sampled = random.sample(task_loaders, n_tasks_per_ep)
        stats   = meta_learner.run_episode(sampled)

        window_G += stats['loss_G']
        window_D += stats['loss_D']

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                [episode + 1, f"{stats['loss_G']:.6f}", f"{stats['loss_D']:.6f}"])

        if (episode + 1) % log_every == 0:
            avg_G    = window_G / log_every
            avg_D    = window_D / log_every
            elapsed  = time.time() - t_start
            eta_sec  = elapsed / (episode + 1) * (meta_epochs - episode - 1)
            eta_str  = time.strftime('%H:%M:%S', time.gmtime(eta_sec))
            logger.info(
                f"Episode {episode + 1:>4}/{meta_epochs} | "
                f"G={avg_G:.4f}  D={avg_D:.4f} | "
                f"elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))}  ETA={eta_str}"
            )
            window_G = window_D = 0.0

        if (episode + 1) % save_every == 0 or episode == meta_epochs - 1:
            path = os.path.join(run_dir, f'reptile_epoch_{episode + 1}.pth')
            torch.save({'G_state_dict': G.state_dict(),
                        'D_state_dict': D.state_dict()}, path)
            logger.info(f"  checkpoint → {path}")

    # 保存供 train.py resume 使用的最终检查点
    final_path = os.path.join(run_dir, 'reptile_meta_init.pth')
    torch.save({
        'epoch':        0,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G':  {},
        'optimizer_D':  {},
        'avg_loss_G':   0.0,
        'avg_loss_D':   0.0,
        'val_loss':     0.0,
    }, final_path)

    logger.info(f"\n元训练完成！初始化参数 → {final_path}")
    logger.info('下一步：config.yaml 中设置 "resume": true, "resume_path": "%s"' % final_path)
    logger.info("然后运行 python train.py 开始二次训练。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    meta_train(load_config(args.config))
