"""
Reptile 元训练入口：学习对所有试卷笔迹风格都泛化的初始参数。

用法:
    python meta_train.py                     # 使用默认 config.yaml
    python meta_train.py --config my.json

完成后在 config.yaml 中设置：
    "resume": true,
    "resume_path": "./reptile_checkpoints/reptile_meta_init.pth"
再运行 python train.py 进行二次训练。
"""
import argparse
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from data.dataset import EnsExamRealDataset
from losses.losses import EnsExamLoss
from networks.discriminator import Discriminator
from networks.generator import Generator
from tools.reptile import ReptileMetaLearner


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'meta_train_{timestamp}.log')

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
    log_cfg     = cfg.get('logging', {})

    log_dir = log_cfg.get('log_dir', './logs')
    logger  = setup_logger(log_dir)

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if train_cfg['device'] == 'auto' else torch.device(train_cfg['device']))
    logger.info(f"使用设备：{device}")

    # 数据集
    dataset = EnsExamRealDataset(
        data_root      = data_cfg['data_root'],
        img_size       = data_cfg['img_size'],
        is_train       = True,
        overlap        = data_cfg['overlap'],
        mask_threshold = data_cfg['mask_threshold'],
        aug_cfg        = data_cfg.get('augmentation'),
    )

    task_loaders   = build_task_loaders(dataset, train_cfg['batch_size'],
                                        pin_memory=device.type == 'cuda')
    n_tasks_per_ep = reptile_cfg['n_tasks_per_episode']
    logger.info(f"共 {len(task_loaders)} 个 task | 每 episode 采样 {n_tasks_per_ep} 个")

    if len(task_loaders) < n_tasks_per_ep:
        raise ValueError(
            f"可用 task 数 ({len(task_loaders)}) < n_tasks_per_episode ({n_tasks_per_ep})"
        )

    # 模型
    G         = Generator(cfg=cfg['model']).to(device)
    D         = Discriminator().to(device)
    criterion = EnsExamLoss(cfg=cfg['loss']).to(device)

    meta_learner = ReptileMetaLearner(G, D, criterion, device, cfg)

    save_dir   = reptile_cfg.get('save_dir', './reptile_checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    meta_epochs = reptile_cfg['meta_epochs']
    save_every  = reptile_cfg.get('save_every_n_epochs', 10)

    # 元训练循环
    for epoch in tqdm(range(meta_epochs), desc='Meta-epochs', ncols=90):
        sampled = random.sample(task_loaders, n_tasks_per_ep)
        meta_learner.run_episode(sampled)

        if (epoch + 1) % save_every == 0 or epoch == meta_epochs - 1:
            path = os.path.join(save_dir, f'reptile_epoch_{epoch + 1}.pth')
            torch.save({'G_state_dict': G.state_dict(),
                        'D_state_dict': D.state_dict()}, path)
            logger.info(f"Epoch {epoch + 1:>4}/{meta_epochs}  checkpoint → {path}")

    # 保存供 train.py resume 使用的最终检查点
    final_path = os.path.join(save_dir, 'reptile_meta_init.pth')
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
