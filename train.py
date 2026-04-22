"""
训练入口：从 config.yaml 加载所有超参数，组装数据集、模型、损失函数并启动训练。

用法（单卡）:
    python train.py                        # 使用默认 config.yaml
    python train.py --config my_cfg.yaml   # 使用自定义配置

用法（多卡 DDP）:
    torchrun --nproc_per_node=2 train.py
    torchrun --nproc_per_node=2 train.py --config my_cfg.yaml
"""
import argparse
import csv
import logging
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch import optim
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from config_loader import load_config, save_config
from data.dataset import EnsExamRealDataset
from losses.losses import EnsExamLoss
from networks.discriminator import Discriminator
from networks.generator import Generator
from utils.eval_metrics import (
    compute_batch_metric_sums,
    finalize_metric_sums,
    format_metric_block,
    init_metric_sums,
    merge_metric_sums,
    paper_display_metrics,
    to_unit_interval,
)
from utils.page_eval import evaluate_full_pages
from utils.path_utils import normalize_path

sys.path.insert(0, os.path.dirname(__file__))
from tools.early_stopping import EarlyStopping


def set_seed(seed: int, mode: str = 'statistical'):
    """固定随机源；支持 strict（严格可复现）与 statistical（统计可复现，默认更快）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if mode == 'strict':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 统计可复现：固定种子 + 允许 cudnn autotune，速度明显更快
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id: int):
    """DataLoader worker 种子固定，确保数据增强顺序可复现。"""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def setup_device(train_cfg: dict):
    """
    解析 gpu_ids 配置，返回 (primary_device, gpu_ids_list)。
    在 DDP 模式下（torchrun 启动），LOCAL_RANK 环境变量会覆盖 gpu_ids。
    """
    if not torch.cuda.is_available():
        return torch.device('cpu'), []

    # DDP 模式：torchrun 会设置 LOCAL_RANK
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        idx = int(local_rank)
        torch.cuda.set_device(idx)
        return torch.device(f'cuda:{idx}'), [idx]

    gpu_ids = train_cfg.get('gpu_ids', None)
    if gpu_ids:
        ids = [int(i) for i in gpu_ids]
        return torch.device(f'cuda:{ids[0]}'), ids

    # 退回到 device 字段
    device_str = train_cfg.get('device', 'auto')
    if device_str == 'auto':
        return torch.device('cuda:0'), [0]
    idx = int(device_str.split(':')[1]) if ':' in device_str else 0
    return torch.device(device_str), [idx]


def is_ddp() -> bool:
    """判断当前是否处于 DDP 模式（由 torchrun 启动）。"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """返回当前进程的全局 rank，非 DDP 时返回 0。"""
    return dist.get_rank() if is_ddp() else 0


def get_world_size() -> int:
    """返回总进程数，非 DDP 时返回 1。"""
    return dist.get_world_size() if is_ddp() else 1


def is_main_process() -> bool:
    """只有 rank 0 做日志/保存/W&B。"""
    return get_rank() == 0


def setup_ddp():
    """初始化 DDP 进程组（仅在 torchrun 环境下调用）。"""
    if os.environ.get('LOCAL_RANK') is not None and not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def cleanup_ddp():
    """销毁 DDP 进程组。"""
    if is_ddp():
        dist.destroy_process_group()


def wrap_model(model: nn.Module, gpu_ids: list) -> nn.Module:
    """DDP 模式用 DistributedDataParallel，否则单卡直接返回。"""
    if is_ddp():
        local_rank = int(os.environ['LOCAL_RANK'])
        # find_unused_parameters=False：训练循环已重构，D step 中 G 绕过 DDP，
        #   G step 中 D 用 no_sync()，不再有真正 unused 的参数。
        # gradient_as_bucket_view=True：梯度直接引用 bucket 内存，省一次拷贝。
        # broadcast_buffers=False：不使用 SyncBN，BN running stats 无需跨 rank 同步。
        return DDP(model, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=False,
                   gradient_as_bucket_view=True,
                   broadcast_buffers=False)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """取出 DDP / DataParallel 内层模型，用于 state_dict 的保存与加载。"""
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


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

def setup_logger(run_dir: str) -> logging.Logger:
    """同时输出到控制台和文件，日志写入 run 目录（目录名已含时间戳）。"""
    log_path = os.path.join(run_dir, 'train.log')

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
def validate(G: Generator, val_loader: DataLoader, device: torch.device) -> dict:
    """
    在验证集上计算全量图像质量指标：
      PSNR、MS-SSIM、MSE、L1、AGE、pEPs、pCEPs
    """
    G.eval()
    sums = init_metric_sums()
    n_images = 0

    for Iin, _, _, _, _, _, Igt in val_loader:
        Iin, Igt = Iin.to(device), Igt.to(device)
        *_, Icomp = G(Iin)

        # [-1,1] → [0,1]
        pred = to_unit_interval(Icomp)
        gt = to_unit_interval(Igt)
        merge_metric_sums(sums, compute_batch_metric_sums(pred, gt))
        n_images += pred.shape[0]

    G.train()
    return finalize_metric_sums(sums, n_images)


# ── 主训练函数 ─────────────────────────────────────────────────────────────────

def train_ensexam(cfg: dict, run_dir: str = None, phase: str = 'train') -> float:
    # DDP 初始化（torchrun 启动时生效，普通 python 启动时跳过）
    setup_ddp()

    train_cfg = cfg['train']
    data_cfg  = cfg['data']
    eval_cfg  = cfg.get('evaluation', {})
    es_cfg    = cfg.get('early_stopping', {})

    epochs      = train_cfg['epochs']
    batch_size  = train_cfg['batch_size']
    lr          = train_cfg['lr']
    adam_betas  = tuple(train_cfg['adam_betas'])
    resume      = train_cfg['resume']
    resume_path = normalize_path(train_cfg['resume_path']) if train_cfg.get('resume_path') else ''
    num_workers = train_cfg['num_workers']
    # Linux 服务器上 num_workers=0 会成为数据加载瓶颈，自动提升
    if num_workers == 0 and os.name != 'nt':
        num_workers = min(4, os.cpu_count() or 1)
    save_every  = train_cfg['save_every_n_epochs']

    data_root      = data_cfg['data_root']
    img_size       = data_cfg['img_size']
    overlap        = data_cfg['overlap']
    mask_threshold = data_cfg['mask_threshold']
    final_test_mode = eval_cfg.get('final_test_mode', 'both')
    page_overlap = int(eval_cfg.get('page_overlap', 32))
    if final_test_mode not in {'patch', 'page', 'both'}:
        raise ValueError(f'未知 final_test_mode: {final_test_mode}')

    # 创建本次运行目录（权重 / 日志 / config 快照统一存放）
    if run_dir is None:
        base_dir  = train_cfg.get('save_dir', './checkpoints')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir   = os.path.join(base_dir, 'ensexam', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, 'config.yaml'))

    logger  = setup_logger(run_dir) if is_main_process() else logging.getLogger('train')
    csv_log = CSVLogger(run_dir) if is_main_process() else None
    wb_run  = init_wandb(cfg) if is_main_process() else None
    wb_cfg  = cfg.get('wandb', {})
    log_img_every = wb_cfg.get('log_image_every_n_epochs', 5)

    seed = train_cfg.get('seed', None)
    reproducibility_mode = train_cfg.get('reproducibility_mode', 'statistical')
    if seed is not None:
        set_seed(seed, mode=reproducibility_mode)
        logger.info(f"随机种子已固定：{seed}（mode={reproducibility_mode}）")

    device, gpu_ids = setup_device(train_cfg)
    if is_main_process():
        if is_ddp():
            logger.info(f"DDP 多卡训练：world_size={get_world_size()}，主设备 {device}")
        else:
            logger.info(f"使用设备：{device}")

    # 数据集
    val_ratio = data_cfg.get('val_ratio', 0.0)
    if val_ratio > 0:
        # 按图像粒度从训练目录划分验证集，避免 patch 级泄漏
        train_img_dir = os.path.join(data_root, 'train', 'all_images')
        valid_ext = ('.png', '.jpg', '.jpeg')
        all_train_files = sorted(
            f for f in os.listdir(train_img_dir) if f.endswith(valid_ext)
        )
        rng = random.Random(seed if seed is not None else 42)
        rng.shuffle(all_train_files)
        n_val = max(1, int(len(all_train_files) * val_ratio))
        val_files   = all_train_files[:n_val]
        train_files = all_train_files[n_val:]
        logger.info(
            f"验证集从训练目录划分：共 {len(all_train_files)} 张图，"
            f"训练 {len(train_files)} 张 / 验证 {len(val_files)} 张（val_ratio={val_ratio}）"
        )
        train_dataset = EnsExamRealDataset(
            data_root=data_root, img_size=img_size, is_train=True,
            overlap=overlap, mask_threshold=mask_threshold,
            aug_cfg=data_cfg.get('augmentation'), file_list=train_files, phase=phase,
        )
        val_dataset = EnsExamRealDataset(
            data_root=data_root, img_size=img_size, is_train=True,
            overlap=0, mask_threshold=mask_threshold,
            aug_cfg=None, file_list=val_files, phase='val',
        )
    else:
        # val_ratio=0：沿用旧行为，直接使用 test 目录
        train_dataset = EnsExamRealDataset(
            data_root=data_root, img_size=img_size, is_train=True,
            overlap=overlap, mask_threshold=mask_threshold,
            aug_cfg=data_cfg.get('augmentation'), phase=phase,
        )
        val_dataset = EnsExamRealDataset(
            data_root=data_root, img_size=img_size, is_train=False,
            overlap=0, mask_threshold=mask_threshold, aug_cfg=None, phase='val',
        )

    pin = device.type == 'cuda'
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    # DDP 模式下使用 DistributedSampler 分片数据
    train_sampler = None
    val_sampler   = None
    if is_ddp():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed or 0)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=num_workers,
                              drop_last=True, pin_memory=pin,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=(2 if num_workers > 0 else None),
                              worker_init_fn=_worker_init_fn, generator=g)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, sampler=val_sampler,
                              num_workers=num_workers,
                              drop_last=False, pin_memory=pin,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=(2 if num_workers > 0 else None))
    if is_main_process():
        logger.info(f"训练集：{len(train_dataset)} patches | 验证集：{len(val_dataset)} patches")

    # 模型
    G = Generator(cfg=cfg['model']).to(device)
    D = Discriminator().to(device)
    criterion = EnsExamLoss(cfg=cfg['loss']).to(device)

    # 注意：不使用 SyncBatchNorm。
    # batch_size=8/GPU 足够 BN 统计，而 SyncBN 在每个 BN 层都做 allreduce，
    # 对本模型（20+ BN 层）开销远大于收益，实测比单卡还慢。

    # optimizer 在 wrap 前创建，持有原始参数引用
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=adam_betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=adam_betas)

    # 学习率调度器
    sch_cfg = train_cfg.get('scheduler', {})
    scheduler_G = scheduler_D = None
    if sch_cfg.get('enabled', False):
        eta_min  = sch_cfg.get('eta_min', 1e-6)
        sch_type = sch_cfg.get('type', 'cosine')
        def _make_scheduler(opt):
            if sch_type == 'cosine_restart':
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt, T_0=sch_cfg.get('T_0', 20),
                    T_mult=sch_cfg.get('T_mult', 2), eta_min=eta_min)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=eta_min)
        scheduler_G = _make_scheduler(optimizer_G)
        scheduler_D = _make_scheduler(optimizer_D)
        logger.info(f"调度器已启用：type={sch_type}, eta_min={eta_min}")

    # 早停
    es = None
    if es_cfg.get('enabled', False):
        es = EarlyStopping(
            patience=es_cfg.get('patience', 10),
            min_delta=es_cfg.get('min_delta', 1e-4),
            mode=es_cfg.get('mode', 'min'),
        )
        logger.info(f"早停已启用：patience={es.patience}, min_delta={es.min_delta}")

    # 断点续训（在 DDP wrap 之前加载，避免 module. 前缀问题）
    start_epoch = 0
    if resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        G.load_state_dict(ckpt['G_state_dict'])
        D.load_state_dict(ckpt['D_state_dict'])
        if ckpt.get('optimizer_G'):
            optimizer_G.load_state_dict(ckpt['optimizer_G'])
        if ckpt.get('optimizer_D'):
            optimizer_D.load_state_dict(ckpt['optimizer_D'])
        if ckpt.get('scheduler_G') and scheduler_G is not None:
            scheduler_G.load_state_dict(ckpt['scheduler_G'])
        if ckpt.get('scheduler_D') and scheduler_D is not None:
            scheduler_D.load_state_dict(ckpt['scheduler_D'])
        start_epoch = ckpt['epoch']
        if is_main_process():
            logger.info(f"断点续训：从第 {start_epoch} epoch 恢复")

    # DDP wrap（checkpoint 加载完成后再 wrap）
    # 只有 G 用 DDP 包裹——GAN 交替训练中，D step 只需 D backward，
    # G step 只需 G gradient sync。若两者都 DDP wrap，必须处理 unused params。
    # D 的梯度同步改为手动 all-reduce，彻底避免 find_unused_parameters 的问题。
    G = wrap_model(G, gpu_ids)
    # D 不包裹 DDP

    # 训练循环
    train_prefetcher = CUDAPrefetcher(train_loader, device)
    best_val_loss = float('inf')
    G.train(); D.train()

    # AMP：A800 等 Ampere+ GPU 使用 bf16 混合精度大幅提升吞吐
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float16
    scaler_G = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    scaler_D = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if is_main_process() and use_amp:
        logger.info(f"AMP 已启用：dtype={amp_dtype}")

    for epoch in range(start_epoch, epochs):
        # DDP：每个 epoch 更新 sampler 的随机种子，保证数据打散
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # GPU 上累加损失，避免每 step .item() 导致的 CPU-GPU 同步
        sum_loss_G = torch.zeros(1, device=device)
        sum_loss_D = torch.zeros(1, device=device)
        sum_parts  = torch.zeros(6, device=device)
        n_steps = 0

        pbar = tqdm(train_prefetcher, total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{epochs}", ncols=100,
                    disable=not is_main_process())

        for Iin, Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt in pbar:
            gt   = (Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt)

            # ── 训练判别器 ──────────────────────────────────────────
            # D 不用 DDP 包裹，手动 all-reduce 梯度（避免 GAN 交替训练的 unused params 问题）
            optimizer_D.zero_grad()
            with torch.no_grad():
                gen_out = unwrap_model(G)(Iin)
            Icomp = gen_out[-1]
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                real_g, real_l = D(Igt,   Mb_gt)
                fake_g, fake_l = D(Icomp, Mb_gt)
                loss_D = (EnsExamLoss.hinge_loss_D(real_g, fake_g)
                          + EnsExamLoss.hinge_loss_D(real_l, fake_l)) / 2
            loss_D.backward()
            if is_ddp():
                ws = get_world_size()
                for p in D.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(ws)
            optimizer_D.step()

            # ── 训练生成器 ──────────────────────────────────────────
            # G 通过 DDP 正常 forward/backward（DDP 自动 allreduce G 的梯度）
            # D 不在 DDP 中，backward 流过 D 但 D 的梯度不需要同步
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                gen_out = G(Iin)
                Icomp   = gen_out[-1]
                fake_g, fake_l = D(Icomp, Mb_gt)
                loss_G, parts  = criterion(gen_out, gt, (fake_g, fake_l))
            loss_G.backward()
            optimizer_G.step()

            # GPU 上累加，不触发 sync
            sum_loss_G += loss_G.detach()
            sum_loss_D += loss_D.detach()
            for i in range(len(parts)):
                sum_parts[i] += parts[i].detach()
            n_steps += 1

            # 进度条：每 20 步更新一次（减少 .item() sync 频率）
            if n_steps % 20 == 0:
                pbar.set_postfix({
                    'D':   f"{(sum_loss_D.item() / n_steps):.3f}",
                    'G':   f"{(sum_loss_G.item() / n_steps):.3f}",
                })

        # epoch 平均（此处一次性 .item()，可接受）
        n = max(n_steps, 1)
        avg_G  = sum_loss_G.item() / n
        avg_D  = sum_loss_D.item() / n
        avg_parts = [sum_parts[i].item() / n for i in range(6)]

        # 验证
        val_m = validate(G, val_loader, device)
        best_val_loss = min(best_val_loss, val_m['l1'])

        current_lr = optimizer_G.param_groups[0]['lr']
        val_display = paper_display_metrics(val_m)

        # 只在 rank 0 上做日志 / CSV / W&B / checkpoint
        if is_main_process():
            logger.info(
                f"Epoch {epoch + 1:>4} | "
                f"Train G={avg_G:.4f}  D={avg_D:.4f} | "
                f"adv={avg_parts[0]:.4f}  rec={avg_parts[1]:.4f}  "
                f"per={avg_parts[2]:.4f}  style={avg_parts[3]:.4f} | "
                f"PSNR={val_m['psnr']:.2f}  "
                f"MS-SSIM={val_display['ms_ssim']:.2f}({val_m['ms_ssim']:.4f})  "
                f"MSE={val_display['mse']:.4f}({val_m['mse']:.6f})  "
                f"AGE={val_m['age']:.2f}  "
                f"pEPs={val_display['peps']:.2f}({val_m['peps']:.4f})  "
                f"pCEPs={val_display['pceps']:.2f}({val_m['pceps']:.4f}) | "
                f"LR={current_lr:.2e}"
            )
            csv_log.write(epoch + 1, avg_G, avg_D, avg_parts, val_m['l1'])

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
                    'val/psnr':         val_m['psnr'],
                    'val/ms_ssim':      val_m['ms_ssim'],
                    'val/mse':          val_m['mse'],
                    'val/l1':           val_m['l1'],
                    'val/age':          val_m['age'],
                    'val/peps':         val_m['peps'],
                    'val/pceps':        val_m['pceps'],
                    'val_display/ms_ssim_x100': val_display['ms_ssim'],
                    'val_display/mse_x100':     val_display['mse'],
                    'val_display/peps_x100':    val_display['peps'],
                    'val_display/pceps_x100':   val_display['pceps'],
                    'train/lr':         current_lr,
                }, step=epoch + 1)
                # 每隔 N epoch 上传对比图
                if (epoch + 1) % log_img_every == 0:
                    log_images_to_wandb(G, val_loader, device, epoch + 1)

            # 保存 checkpoint（先存再 step，确保 lr 与本 epoch 对应）
            ckpt = {
                'epoch': epoch + 1,
                'G_state_dict': unwrap_model(G).state_dict(),
                'D_state_dict': unwrap_model(D).state_dict(),
                'optimizer_G':  optimizer_G.state_dict(),
                'optimizer_D':  optimizer_D.state_dict(),
                'scheduler_G':  scheduler_G.state_dict() if scheduler_G else None,
                'scheduler_D':  scheduler_D.state_dict() if scheduler_D else None,
                'avg_loss_G':   avg_G,
                'avg_loss_D':   avg_D,
                'val_loss':     val_m['l1'],
            }
            torch.save(ckpt, os.path.join(run_dir, 'latest.pth'))

        if scheduler_G is not None:
            scheduler_G.step()
            scheduler_D.step()

        if is_main_process():
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                path = os.path.join(run_dir, f'epoch_{epoch + 1}.pth')
                torch.save(ckpt, path)
                logger.info(f"已保存：{path}")

        # 保存最优模型（由早停判定，监控 PSNR）——所有 rank 都 step 以保持同步
        if es is not None:
            should_stop = es.step(val_m['psnr'], epoch + 1)
            if is_main_process():
                if should_stop:
                    logger.info(
                        f"早停触发：连续 {es.patience} epoch 无改善，"
                        f"最优 epoch={es.best_epoch}，PSNR={es.best_value:.2f}"
                    )
                if es.is_best:
                    torch.save(ckpt, os.path.join(run_dir, 'best.pth'))
                    logger.info(f"已更新最优模型 best.pth（PSNR={val_m['psnr']:.2f}）")
                    if wb_run is not None:
                        wandb.run.summary['best_psnr']  = val_m['psnr']
                        wandb.run.summary['best_epoch'] = epoch + 1
            if should_stop:
                break

    # ── 训练结束后，在测试集上评估最优模型 ─────────────────────────────────────
    if is_main_process():
        logger.info("=" * 60)
        logger.info("训练结束，开始在测试集上评估最优模型...")

        # 加载最优权重（best.pth）；若不存在则使用 latest.pth
        best_path   = os.path.join(run_dir, 'best.pth')
        latest_path = os.path.join(run_dir, 'latest.pth')
        eval_ckpt_path = best_path if os.path.exists(best_path) else latest_path

        if os.path.exists(eval_ckpt_path):
            eval_ckpt = torch.load(eval_ckpt_path, map_location=device, weights_only=False)
            unwrap_model(G).load_state_dict(eval_ckpt['G_state_dict'])
            eval_epoch = eval_ckpt.get('epoch', '?')
            logger.info(f"已加载权重：{eval_ckpt_path}（epoch={eval_epoch}）")
        else:
            logger.warning("未找到保存的权重文件，使用当前模型状态进行测试集评估")

        patch_test_m = None
        page_test_m = None
        page_count = 0

        if final_test_mode in ('patch', 'both'):
            test_dataset = EnsExamRealDataset(
                data_root=data_root, img_size=img_size, is_train=False,
                overlap=0, mask_threshold=mask_threshold, aug_cfg=None, phase='test',
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, drop_last=False, pin_memory=pin,
                persistent_workers=(num_workers > 0),
                prefetch_factor=(2 if num_workers > 0 else None),
            )
            logger.info(f"Patch 测试集：{len(test_dataset)} patches")
            patch_test_m = validate(unwrap_model(G), test_loader, device)

        if final_test_mode in ('page', 'both'):
            logger.info(f"Page 测试集：整页评估，overlap={page_overlap}px")
            page_test_m, page_count = evaluate_full_pages(
                unwrap_model(G),
                data_root=data_root,
                device=device,
                phase='test',
                overlap=page_overlap,
            )

        logger.info("-" * 60)
        if patch_test_m is not None:
            logger.info("Patch 级测试集评估结果：")
            for line in format_metric_block(patch_test_m):
                logger.info(line)
        if page_test_m is not None:
            logger.info(f"Page 级测试集评估结果（{page_count} 张整页）：")
            for line in format_metric_block(page_test_m):
                logger.info(line)
        logger.info("=" * 60)

        # W&B 上报测试集指标
        if wb_run is not None:
            wandb_payload = {}
            if patch_test_m is not None:
                test_display = paper_display_metrics(patch_test_m)
                wandb_payload.update({
                    'test/psnr':    patch_test_m['psnr'],
                    'test/ms_ssim': patch_test_m['ms_ssim'],
                    'test/mse':     patch_test_m['mse'],
                    'test/l1':      patch_test_m['l1'],
                    'test/age':     patch_test_m['age'],
                    'test/peps':    patch_test_m['peps'],
                    'test/pceps':   patch_test_m['pceps'],
                    'test_display/ms_ssim_x100': test_display['ms_ssim'],
                    'test_display/mse_x100':     test_display['mse'],
                    'test_display/peps_x100':    test_display['peps'],
                    'test_display/pceps_x100':   test_display['pceps'],
                })
                wandb.run.summary['test_psnr'] = patch_test_m['psnr']
                wandb.run.summary['test_ms_ssim'] = patch_test_m['ms_ssim']
                wandb.run.summary['test_mse'] = patch_test_m['mse']
                wandb.run.summary['test_l1'] = patch_test_m['l1']
                wandb.run.summary['test_age'] = patch_test_m['age']
                wandb.run.summary['test_peps'] = patch_test_m['peps']
                wandb.run.summary['test_pceps'] = patch_test_m['pceps']

            if page_test_m is not None:
                page_display = paper_display_metrics(page_test_m)
                wandb_payload.update({
                    'test_page/psnr':    page_test_m['psnr'],
                    'test_page/ms_ssim': page_test_m['ms_ssim'],
                    'test_page/mse':     page_test_m['mse'],
                    'test_page/l1':      page_test_m['l1'],
                    'test_page/age':     page_test_m['age'],
                    'test_page/peps':    page_test_m['peps'],
                    'test_page/pceps':   page_test_m['pceps'],
                    'test_page_display/ms_ssim_x100': page_display['ms_ssim'],
                    'test_page_display/mse_x100':     page_display['mse'],
                    'test_page_display/peps_x100':    page_display['peps'],
                    'test_page_display/pceps_x100':   page_display['pceps'],
                })
                wandb.run.summary['test_page_psnr'] = page_test_m['psnr']
                wandb.run.summary['test_page_ms_ssim'] = page_test_m['ms_ssim']
                wandb.run.summary['test_page_mse'] = page_test_m['mse']
                wandb.run.summary['test_page_l1'] = page_test_m['l1']
                wandb.run.summary['test_page_age'] = page_test_m['age']
                wandb.run.summary['test_page_peps'] = page_test_m['peps']
                wandb.run.summary['test_page_pceps'] = page_test_m['pceps']

            if wandb_payload:
                wandb.log(wandb_payload)

    if wb_run is not None:
        wandb.finish()
    cleanup_ddp()

    return es.best_value if es is not None else best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='config.yaml 路径')
    args = parser.parse_args()
    train_ensexam(load_config(args.config), phase='train')
