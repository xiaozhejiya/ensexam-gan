"""
Local-Global 判别器：全局判别器输出标量，局部判别器输出特征图并用文本掩码加权。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.blocks import DiscBlock


class Discriminator(nn.Module):
    """Local-Global 判别器。

    Args:
        in_channels: 输入通道数，RGB 固定为 3

    Forward:
        x:          [B, 3, H, W] 归一化到 [-1, 1] 的图像
        local_mask: [B, 1, H, W] 文本块掩码（训练时传入，推理时可省略）

    Returns:
        global_score: [B, 1, 1, 1] 全局判别 logits
        local_score:  [B, 1, H', W'] 掩码加权后的局部判别 logits（无掩码时为 0）
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.global_disc = nn.Sequential(
            DiscBlock(in_channels, 64),
            DiscBlock(64, 128),
            DiscBlock(128, 256),
            DiscBlock(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),   # 输出标量
        )
        self.local_disc = nn.Sequential(
            DiscBlock(in_channels, 64),
            DiscBlock(64, 128),
            DiscBlock(128, 256),
            DiscBlock(256, 512),
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),   # 输出特征图
        )

    def forward(self, x: torch.Tensor, local_mask: torch.Tensor = None):
        global_score = self.global_disc(x)
        local_score = 0

        if local_mask is not None:
            local_feat = self.local_disc(x)
            _, _, h, w = local_feat.shape
            mask_scaled = F.interpolate(local_mask, size=(h, w), mode='nearest')
            local_score = local_feat * mask_scaled

        return global_score, local_score
