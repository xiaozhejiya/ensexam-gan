"""
早停：若监控指标在连续 patience 个 epoch 内无实质性改善，则停止训练。
"""
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Args:
        patience:  无改善时最多再等待的 epoch 数
        min_delta: 认定为"有改善"的最小变化量（绝对值）
        mode:      'min'（越小越好，如 loss）或 'max'（越大越好，如 PSNR）
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode 必须是 'min' 或 'max'，当前值：{mode}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0   # 距上次改善已过的 epoch 数

    def step(self, value: float, epoch: int) -> bool:
        """
        传入当前 epoch 的监控指标值。

        Returns:
            True  → 应停止训练
            False → 继续训练
        """
        improved = (
            value < self.best_value - self.min_delta if self.mode == 'min'
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            logger.info(f"[EarlyStopping] 指标改善 → best={self.best_value:.6f} (epoch {epoch})")
        else:
            self.counter += 1
            logger.info(
                f"[EarlyStopping] 无改善 {self.counter}/{self.patience}，"
                f"best={self.best_value:.6f} (epoch {self.best_epoch})"
            )

        return self.counter >= self.patience

    @property
    def is_best(self) -> bool:
        """当前 epoch 是否是迄今最优。"""
        return self.counter == 0
