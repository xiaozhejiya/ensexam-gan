"""
Optuna 超参数调优：在 Reptile 元初始化基础上，搜索最优微调超参数。

使用方法：
    # 首次启动
    python tune.py

    # 中途中断后续调（SQLite 自动持久化）
    python tune.py --resume

    # 启动 optuna-dashboard 实时监控（另开一个终端）
    optuna-dashboard sqlite:///tuning.db

    # 调优完成后，把最优参数写回 config.yaml，再运行完整训练
    python train.py
"""
import argparse
import copy
import os
import sys

import optuna
import torch

# 确保 models/ 根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from config_loader import load_config
from train import train_ensexam

# 屏蔽 Optuna 的 INFO 日志，避免刷屏
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_storage(storage_url: str):
    """
    优先使用 SQLite（支持 optuna-dashboard）；
    若 _sqlite3 DLL 缺失（Windows 环境常见问题），自动降级到
    JournalFileStorage（同样持久化、支持断点续调，但不支持 dashboard）。
    """
    try:
        import sqlite3  # noqa: F401  — 仅做可用性探针
        return storage_url
    except ImportError:
        log_path = storage_url.replace("sqlite:///", "").replace(".db", ".log")
        print(f"[警告] sqlite3 不可用，降级到 JournalFileStorage → {log_path}")
        print("       如需 optuna-dashboard，请执行：")
        print("       conda install -n wav2_3.7 sqlite --force-reinstall -c conda-forge\n")
        return optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(log_path)
        )


# ── 构建 trial 配置 ────────────────────────────────────────────────────────────

def build_trial_cfg(trial: optuna.Trial, base_cfg: dict, tune_cfg: dict) -> dict:
    """
    用 trial 的采样参数覆盖 base_cfg，返回该 trial 专用的配置副本。
    所有修改只在副本上进行，base_cfg 保持不变。
    """
    cfg = copy.deepcopy(base_cfg)
    ss  = tune_cfg['search_space']

    # 训练超参数
    cfg['train']['lr']            = trial.suggest_float('lr',    *ss['lr'],    log=True)
    cfg['train']['adam_betas'][0] = trial.suggest_float('beta1', *ss['beta1'])

    # 损失函数权重
    cfg['loss']['lambda_p']     = trial.suggest_float('lambda_p',     *ss['lambda_p'],     log=True)
    cfg['loss']['lambda_style'] = trial.suggest_float('lambda_style', *ss['lambda_style'], log=True)
    cfg['loss']['lambda_b']     = trial.suggest_float('lambda_b',     *ss['lambda_b'])

    # 调优专用覆盖：缩短 epoch、从元初始化权重出发
    cfg['train']['epochs']      = tune_cfg['tune_epochs']
    cfg['train']['resume']      = True
    cfg['train']['resume_path'] = tune_cfg['init_weights']

    # 关闭早停（5 epoch 太短，触发无意义）
    cfg['early_stopping']['enabled'] = False

    # W&B 由 tune.py 统一管理，每个 trial 内部不再创建 run
    cfg['wandb']['enabled'] = False

    # 日志写到独立子目录，避免和主训练日志混淆
    cfg['logging']['log_dir'] = os.path.join(
        base_cfg['logging'].get('log_dir', './logs'), 'tuning'
    )
    return cfg


# ── W&B 回调 ──────────────────────────────────────────────────────────────────

def make_wandb_callback(wb_run):
    """
    返回 Optuna callback：每个 trial 完成后向 W&B 上报参数和结果。
    整个调优过程共用同一个 W&B run，方便在参数重要性图里对比所有 trial。
    """
    if wb_run is None:
        return None

    # 用 wandb.Table 积累所有 trial 数据，方便后续筛选
    table = wandb.Table(columns=[
        'trial', 'lr', 'beta1', 'lambda_p', 'lambda_style', 'lambda_b', 'val_loss'
    ])

    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        p = trial.params
        wandb.log({
            'optuna/val_loss':      trial.value,
            'optuna/best_val_loss': study.best_value,
            'optuna/trial_number':  trial.number,
            'optuna/lr':            p['lr'],
            'optuna/beta1':         p['beta1'],
            'optuna/lambda_p':      p['lambda_p'],
            'optuna/lambda_style':  p['lambda_style'],
            'optuna/lambda_b':      p['lambda_b'],
        }, step=trial.number)

        table.add_data(
            trial.number,
            p['lr'], p['beta1'], p['lambda_p'], p['lambda_style'], p['lambda_b'],
            trial.value,
        )
        wandb.log({'optuna/trials_table': table}, step=trial.number)

    return callback


# ── 主调优函数 ─────────────────────────────────────────────────────────────────

def run_tuning(cfg: dict, resume: bool = False):
    tune_cfg = cfg['tuning']
    wb_cfg   = cfg.get('wandb', {})

    # 检查元初始化权重存在
    init_path = tune_cfg['init_weights']
    if not os.path.exists(init_path):
        raise FileNotFoundError(
            f"元初始化权重不存在：{init_path}\n"
            "请先运行 python meta_train.py 完成 Reptile 元训练。"
        )

    print(f"调优起点权重：{init_path}")
    print(f"每 trial epoch 数：{tune_cfg['tune_epochs']}")
    print(f"计划 trial 总数：{tune_cfg['n_trials']}")
    print(f"SQLite 存储：{tune_cfg['storage']}\n")

    # W&B：整个调优过程共用一个 run
    wb_run = None
    if wb_cfg.get('enabled', False) and _WANDB_AVAILABLE:
        wb_run = wandb.init(
            project=wb_cfg.get('project', 'ensexam'),
            name=f"tune-{tune_cfg['study_name']}",
            group='hyperparameter-tuning',
            config={
                'search_space': tune_cfg['search_space'],
                'tune_epochs':  tune_cfg['tune_epochs'],
                'n_trials':     tune_cfg['n_trials'],
                'init_weights': init_path,
            },
        )
        print(f"W&B run：{wb_run.url}\n")

    # Objective
    def objective(trial: optuna.Trial) -> float:
        trial_cfg = build_trial_cfg(trial, cfg, tune_cfg)
        print(f"\n[Trial {trial.number}] "
              f"lr={trial.params['lr']:.2e}  "
              f"beta1={trial.params['beta1']:.3f}  "
              f"lambda_p={trial.params['lambda_p']:.3f}  "
              f"lambda_style={trial.params['lambda_style']:.1f}  "
              f"lambda_b={trial.params['lambda_b']:.3f}")
        return train_ensexam(trial_cfg)

    # 创建或恢复 study
    storage    = _build_storage(tune_cfg['storage'])
    study_name = tune_cfg['study_name']
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,     # resume=True 时续调，False 时若已存在也不报错
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # 前 5 个 trial 随机采样，不剪枝
            n_warmup_steps=2,    # 每个 trial 至少跑 2 epoch 后才考虑剪枝
        ),
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    if resume and completed > 0:
        print(f"续调：已完成 {completed} 个 trial，继续运行。")

    callbacks = [c for c in [make_wandb_callback(wb_run)] if c is not None]
    study.optimize(objective, n_trials=tune_cfg['n_trials'], callbacks=callbacks)

    # 打印最优结果
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"  调优完成  最优 trial #{best.number}")
    print(f"  val_loss  = {best.value:.6f}")
    print("  最优超参数：")
    print(f"    lr            = {best.params['lr']:.2e}")
    print(f"    adam_betas[0] = {best.params['beta1']:.4f}")
    print(f"    lambda_p      = {best.params['lambda_p']:.4f}")
    print(f"    lambda_style  = {best.params['lambda_style']:.2f}")
    print(f"    lambda_b      = {best.params['lambda_b']:.4f}")
    print("=" * 60)
    print("\n将以上参数更新到 config.yaml 后，运行完整训练：")
    print("  python train.py")

    if wb_run is not None:
        wandb.run.summary['best_trial']    = best.number
        wandb.run.summary['best_val_loss'] = best.value
        wandb.run.summary.update(best.params)
        wandb.finish()

    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='config.yaml 路径')
    parser.add_argument('--resume', action='store_true',   help='续调上次中断的 study')
    args = parser.parse_args()
    run_tuning(load_config(args.config), resume=args.resume)
