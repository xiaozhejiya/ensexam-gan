"""
可视化学习率曲线，用于检查余弦退火调度是否按预期运行。

用法：
    python tools/plot_lr.py                        # 弹窗显示
    python tools/plot_lr.py --output lr_curve.png  # 保存图片
    python tools/plot_lr.py --config my_cfg.yaml   # 指定配置文件
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config_loader import load_config


def plot_lr_schedule(cfg: dict, output: str = None):
    train_cfg = cfg['train']
    sch_cfg   = train_cfg.get('scheduler', {})

    if not sch_cfg.get('enabled', False):
        print("调度器未启用（scheduler.enabled=false），请先在 config.yaml 中开启。")
        return

    lr       = train_cfg['lr']
    epochs   = train_cfg['epochs']
    eta_min  = sch_cfg.get('eta_min', 1e-6)
    sch_type = sch_cfg.get('type', 'cosine')

    dummy = torch.nn.Linear(1, 1)
    opt   = optim.Adam(dummy.parameters(), lr=lr)

    if sch_type == 'cosine_restart':
        T_0    = sch_cfg.get('T_0', 20)
        T_mult = sch_cfg.get('T_mult', 2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        subtitle = f"CosineAnnealingWarmRestarts   T_0={T_0}  T_mult={T_mult}  eta_min={eta_min:.0e}"
        # 标出每次重启的位置
        restarts = []
        t = T_0
        while t < epochs:
            restarts.append(t)
            t += T_0 * (T_mult ** len(restarts))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=eta_min)
        subtitle  = f"CosineAnnealingLR   T_max={epochs}  eta_min={eta_min:.0e}"
        restarts  = []

    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]['lr'])
        scheduler.step()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(1, epochs + 1), lrs, color='steelblue', linewidth=1.8, label='LR (G / D)')
    ax.axhline(eta_min, color='gray', linestyle='--', linewidth=0.9, label=f'eta_min = {eta_min:.0e}')
    ax.axhline(lr,      color='salmon', linestyle='--', linewidth=0.9, label=f'lr_init = {lr:.0e}')

    for r in restarts:
        ax.axvline(r, color='orange', linestyle=':', linewidth=0.9, alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(f'Learning Rate Schedule\n{subtitle}', fontsize=12)
    ax.set_xlim(1, epochs)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"已保存：{output}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='config.yaml 路径')
    parser.add_argument('--output', default=None,          help='图片保存路径（不指定则弹窗显示）')
    args = parser.parse_args()
    plot_lr_schedule(load_config(args.config), args.output)
