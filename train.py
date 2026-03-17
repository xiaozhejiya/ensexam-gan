"""
训练入口：从 config.yaml 加载所有超参数，组装数据集、模型、损失函数并启动训练。

用法:
    python train.py                        # 使用默认 config.yaml
    python train.py --config my_cfg.yaml   # 使用自定义配置
"""
import argparse
import csv
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from config_loader import load_config
from data.dataset import EnsExamRealDataset
from losses.losses import EnsExamLoss
from networks.discriminator import Discriminator
from networks.generator import Generator

sys.path.insert(0, os.path.dirname(__file__))
from tools.early_stopping import EarlyStopping


class CUDAPrefetcher:
    """用独立 CUDA Stream 将下一批数据的 H2D 传输与当前批 GPU 计算并行。"""

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None

    def __iter__(self):
        it = iter(self.loader)
        batch = next(it, None)
        if batch is None:
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                batch = [t.to(self.device, non_blocking=True) for t in batch]

        while True:
            try:
                next_batch = next(it)
            except StopIteration:
                next_batch = None

            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)

            yield batch

            if next_batch is None:
                break

            batch = next_batch
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    batch = [t.to(self.device, non_blocking=True) for t in batch]

    def __len__(self):
        return len(self.loader)


# ── 日志初始化 ─────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str) -> logging.Logger:
    """同时输出到控制台和文件，文件名含时间戳避免覆盖。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'train_{timestamp}.log')

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s',
                             datefmt='%H:%M:%S')
    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # 文件
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"日志文件：{log_path}")
    return logger


class CSVLogger:
    """每个 epoch 追加一行到 loss_history.csv，方便后续绘图。"""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, 'loss_history.csv')
        # 若文件不存在则写表头
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(
                    ['epoch', 'train_G', 'train_D',
                     'train_adv', 'train_lr', 'train_per', 'train_style',
                     'val_loss']
                )

    def write(self, epoch: int, train_G: float, train_D: float,
              parts: list, val_loss: float):
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, f'{train_G:.6f}', f'{train_D:.6f}',
                f'{parts[0]:.6f}', f'{parts[1]:.6f}',
                f'{parts[2]:.6f}', f'{parts[3]:.6f}',
                f'{val_loss:.6f}',
            ])


# ── W&B 工具 ───────────────────────────────────────────────────────────────────

def init_wandb(cfg: dict):
    """初始化 W&B run，把完整 config 上传作为超参数记录。返回 run 对象或 None。"""
    wb_cfg = cfg.get('wandb', {})
    if not wb_cfg.get('enabled', False) or not _WANDB_AVAILABLE:
        return None
    run = wandb.init(
        project=wb_cfg.get('project', 'ensexam'),
        name=wb_cfg.get('run_name') or None,   # None = W&B 自动命名
        config=cfg,                             # 把整个 config.yaml 存入 W&B
        resume='allow',
    )
    return run


@torch.no_grad()
def log_images_to_wandb(G, val_loader, device, epoch):
    """从验证集取一批样本，上传 Iin / Icomp / Igt 对比图到 W&B。"""
    G.eval()
    Iin, _, Mb_gt, _, _, _, Igt = next(iter(val_loader))
    Iin, Igt = Iin.to(device), Igt.to(device)
    *_, Icomp = G(Iin)

    def to_uint8(t):
        """[-1, 1] tensor → (H, W, 3) uint8 numpy，取第一张。"""
        img = t[0].cpu().float()
        img = ((img + 1) / 2).clamp(0, 1)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    wandb.log({
        'samples/input':  wandb.Image(to_uint8(Iin),   caption='输入（含笔记）'),
        'samples/output': wandb.Image(to_uint8(Icomp), caption='擦除结果'),
        'samples/gt':     wandb.Image(to_uint8(Igt),   caption='GT（干净底图）'),
    }, step=epoch)
    G.train()


# ── 验证循环 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(G: Generator, val_loader: DataLoader, device: torch.device) -> float:
    """
    在验证集上计算平均重建损失（L1），不含 GAN 对抗项。
    用 L1(Icomp, Igt) 作为早停指标：纯重建质量，不受判别器训练状态影响。
    """
    G.eval()
    total_loss = 0.0
    for Iin, _, Mb_gt, _, _, _, Igt in val_loader:
        Iin, Mb_gt, Igt = Iin.to(device), Mb_gt.to(device), Igt.to(device)
        *_, Icomp = G(Iin)
        # 文本区域 L1 × 2 + 背景区域 L1（更关注笔画擦除质量）
        total_loss += (2 * F.l1_loss(Icomp * Mb_gt, Igt * Mb_gt)
                       + F.l1_loss(Icomp * (1 - Mb_gt), Igt * (1 - Mb_gt))).item()
    G.train()
    return total_loss / len(val_loader)


# ── 主训练函数 ─────────────────────────────────────────────────────────────────

def train_ensexam(cfg: dict):
    train_cfg = cfg['train']
    data_cfg  = cfg['data']
    es_cfg    = cfg.get('early_stopping', {})
    log_cfg   = cfg.get('logging', {})

    epochs      = train_cfg['epochs']
    batch_size  = train_cfg['batch_size']
    lr          = train_cfg['lr']
    adam_betas  = tuple(train_cfg['adam_betas'])
    save_dir    = train_cfg['save_dir']
    resume      = train_cfg['resume']
    resume_path = train_cfg['resume_path']
    num_workers = train_cfg['num_workers']
    save_every  = train_cfg['save_every_n_epochs']

    data_root      = data_cfg['data_root']
    img_size       = data_cfg['img_size']
    overlap        = data_cfg['overlap']
    mask_threshold = data_cfg['mask_threshold']

    log_dir = log_cfg.get('log_dir', './logs')
    logger  = setup_logger(log_dir)
    csv_log = CSVLogger(log_dir)
    wb_run  = init_wandb(cfg)
    wb_cfg  = cfg.get('wandb', {})
    log_img_every = wb_cfg.get('log_image_every_n_epochs', 5)

    device_str = train_cfg['device']
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))
    logger.info(f"使用设备：{device}")

    os.makedirs(save_dir, exist_ok=True)

    # 数据集
    train_dataset = EnsExamRealDataset(
        data_root=data_root, img_size=img_size, is_train=True,
        overlap=overlap, mask_threshold=mask_threshold,
        aug_cfg=data_cfg.get('augmentation'),
    )
    val_dataset = EnsExamRealDataset(
        data_root=data_root, img_size=img_size, is_train=False,
        overlap=0, mask_threshold=mask_threshold, aug_cfg=None,
    )
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              drop_last=True, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              drop_last=False, pin_memory=pin)
    logger.info(f"训练集：{len(train_dataset)} patches | 验证集：{len(val_dataset)} patches")

    # 模型
    G = Generator(cfg=cfg['model']).to(device)
    D = Discriminator().to(device)
    criterion = EnsExamLoss(cfg=cfg['loss']).to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=adam_betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=adam_betas)

    # 早停
    es = None
    if es_cfg.get('enabled', False):
        es = EarlyStopping(
            patience=es_cfg.get('patience', 10),
            min_delta=es_cfg.get('min_delta', 1e-4),
            mode=es_cfg.get('mode', 'min'),
        )
        logger.info(f"早停已启用：patience={es.patience}, min_delta={es.min_delta}")

    # 断点续训
    start_epoch = 0
    if resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        G.load_state_dict(ckpt['G_state_dict'])
        D.load_state_dict(ckpt['D_state_dict'])
        # reptile_meta_init.pth 的 optimizer 为空 dict，跳过以避免报错
        if ckpt.get('optimizer_G'):
            optimizer_G.load_state_dict(ckpt['optimizer_G'])
        if ckpt.get('optimizer_D'):
            optimizer_D.load_state_dict(ckpt['optimizer_D'])
        start_epoch = ckpt['epoch']
        logger.info(f"断点续训：从第 {start_epoch} epoch 恢复")

    # 训练循环
    train_prefetcher = CUDAPrefetcher(train_loader, device)
    best_val_loss = float('inf')
    G.train(); D.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss_G = epoch_loss_D = 0.0
        epoch_parts  = [0.0] * 6   # [adv, lr, per, style, sn, block]
        pbar = tqdm(train_prefetcher, total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{epochs}", ncols=100)

        for Iin, Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt in pbar:
            gt   = (Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt)

            # 训练判别器
            optimizer_D.zero_grad()
            gen_out = G(Iin)
            Icomp   = gen_out[-1]
            real_g, real_l = D(Igt, Mb_gt)
            fake_g, fake_l = D(Icomp.detach(), Mb_gt)
            loss_D = (EnsExamLoss.hinge_loss_D(real_g, fake_g)
                      + EnsExamLoss.hinge_loss_D(real_l, fake_l)) / 2
            loss_D.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            fake_g, fake_l = D(Icomp, Mb_gt)
            loss_G, parts  = criterion(gen_out, gt, (fake_g, fake_l))
            loss_G.backward()
            optimizer_G.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            for i, p in enumerate(parts):
                epoch_parts[i] += p.item()

            pbar.set_postfix({
                'D':   f"{loss_D.item():.3f}",
                'G':   f"{loss_G.item():.3f}",
                'adv': f"{parts[0].item():.3f}",
                'lr':  f"{parts[1].item():.3f}",
            })

        # epoch 平均
        n = len(train_loader)
        avg_G  = epoch_loss_G / n
        avg_D  = epoch_loss_D / n
        avg_parts = [p / n for p in epoch_parts]

        # 验证
        val_loss = validate(G, val_loader, device)
        best_val_loss = min(best_val_loss, val_loss)

        logger.info(
            f"Epoch {epoch + 1:>4} | "
            f"Train G={avg_G:.4f}  D={avg_D:.4f} | "
            f"adv={avg_parts[0]:.4f}  lr={avg_parts[1]:.4f}  "
            f"per={avg_parts[2]:.4f}  style={avg_parts[3]:.4f} | "
            f"Val L1={val_loss:.4f}"
        )
        csv_log.write(epoch + 1, avg_G, avg_D, avg_parts, val_loss)

        # W&B：上报数值指标
        if wb_run is not None:
            wandb.log({
                'train/loss_G':     avg_G,
                'train/loss_D':     avg_D,
                'train/adv':        avg_parts[0],
                'train/lr_loss':    avg_parts[1],
                'train/perceptual': avg_parts[2],
                'train/style':      avg_parts[3],
                'train/sn':         avg_parts[4],
                'train/block':      avg_parts[5],
                'val/l1_loss':      val_loss,
            }, step=epoch + 1)
            # 每隔 N epoch 上传对比图
            if (epoch + 1) % log_img_every == 0:
                log_images_to_wandb(G, val_loader, device, epoch + 1)

        # 保存 checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'optimizer_G':  optimizer_G.state_dict(),
            'optimizer_D':  optimizer_D.state_dict(),
            'avg_loss_G':   avg_G,
            'avg_loss_D':   avg_D,
            'val_loss':     val_loss,
        }
        torch.save(ckpt, os.path.join(save_dir, 'latest.pth'))
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
            torch.save(ckpt, path)
            logger.info(f"已保存：{path}")

        # 保存最优模型（由早停判定）
        if es is not None:
            if es.is_best:
                torch.save(ckpt, os.path.join(save_dir, 'best.pth'))
                logger.info(f"已更新最优模型 best.pth（val_loss={val_loss:.4f}）")
                if wb_run is not None:
                    wandb.run.summary['best_val_loss'] = val_loss
                    wandb.run.summary['best_epoch']    = epoch + 1
            if es.step(val_loss, epoch + 1):
                logger.info(
                    f"早停触发：连续 {es.patience} epoch 无改善，"
                    f"最优 epoch={es.best_epoch}，val_loss={es.best_value:.4f}"
                )
                break

    if wb_run is not None:
        wandb.finish()

    return best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='config.yaml 路径')
    args = parser.parse_args()
    train_ensexam(load_config(args.config))
