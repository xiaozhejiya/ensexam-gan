"""
生成器网络：CoarseNet（粗擦除 + 掩码预测） + RefineNet（精细修复）。
"""
import torch
import torch.nn as nn

from networks.blocks import DownSample, UpSample, DilatedUpSample


class CoarseNet(nn.Module):
    """粗擦除网络：U-Net 结构，双解码器分别输出多尺度修复图和掩码。

    Args:
        in_channels: 输入通道数，RGB 固定为 3
        cbam_reduction: CBAM 通道压缩比
    """
    def __init__(self, in_channels: int = 3, cbam_reduction: int = 16):
        super().__init__()
        # 编码器
        self.down1 = DownSample(in_channels, 64)   # H/2
        self.down2 = DownSample(64, 128)            # H/4
        self.down3 = DownSample(128, 256)           # H/8
        self.down4 = DownSample(256, 512)           # H/16
        self.down5 = DownSample(512, 512)           # H/32

        # 修复解码器（输出多尺度 Ic）
        self.up1 = UpSample(512, 512, use_cbam=True)
        self.up2 = UpSample(1024, 256, use_cbam=True)
        self.up3 = UpSample(512, 128, use_cbam=True)   # → Ic4
        self.up4 = UpSample(256, 64, use_cbam=True)    # → Ic2
        self.up5 = UpSample(128, 64, use_cbam=True)    # → Ic1

        # 掩码解码器（输出 Ms、Mb）
        self.up1_seg = UpSample(512, 512, use_cbam=True)
        self.up2_seg = UpSample(1024, 256, use_cbam=True)
        self.up3_seg = UpSample(512, 128, use_cbam=True)
        self.up4_seg = UpSample(256, 64, use_cbam=True)
        self.up5_seg = UpSample(128, 64, use_cbam=True)

        self.out_ms  = nn.Conv2d(64, 1, 3, 1, 1)        # 软笔画掩码
        self.out_mb  = nn.Conv2d(64, 1, 3, 1, 1)        # 文本块掩码
        self.out_ic4 = nn.Conv2d(128, 3, 3, 1, 1, bias=False)  # 1/4 修复图
        self.out_ic2 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)   # 1/2 修复图
        self.out_ic1 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)   # 1/1 修复图

    def forward(self, x: torch.Tensor):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # 修复分支
        u1 = torch.cat([self.up1(d5), d4], dim=1)
        u2 = torch.cat([self.up2(u1), d3], dim=1)
        u3 = self.up3(u2)
        Ic4 = torch.tanh(self.out_ic4(u3))
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.up4(u3)
        Ic2 = torch.tanh(self.out_ic2(u4))
        u4 = torch.cat([u4, d1], dim=1)
        u5 = self.up5(u4)
        Ic1 = torch.tanh(self.out_ic1(u5))

        # 掩码分支
        s1 = torch.cat([self.up1_seg(d5), d4], dim=1)
        s2 = torch.cat([self.up2_seg(s1), d3], dim=1)
        s3 = torch.cat([self.up3_seg(s2), d2], dim=1)
        s4 = torch.cat([self.up4_seg(s3), d1], dim=1)
        s5 = self.up5_seg(s4)
        Ms = torch.sigmoid(self.out_ms(s5))
        Mb = torch.sigmoid(self.out_mb(s5))

        return Ms, Mb, Ic4, Ic2, Ic1


class RefineNet(nn.Module):
    """精细修复网络：以 CoarseNet 输出为输入，用空洞卷积扩大感受野做精细还原。

    Args:
        in_channels: 输入通道数 = Iin(3) + Ms(1) + Ic1(3) = 7
    """
    def __init__(self, in_channels: int = 7):
        super().__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.up1 = DilatedUpSample(512, 256)
        self.up2 = DilatedUpSample(512, 128)
        self.up3 = DilatedUpSample(256, 64)
        self.up4 = DilatedUpSample(128, 64)

        self.out_ire = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = torch.cat([self.up1(d4), d3], dim=1)
        u2 = torch.cat([self.up2(u1), d2], dim=1)
        u3 = torch.cat([self.up3(u2), d1], dim=1)
        u4 = self.up4(u3)

        return torch.tanh(self.out_ire(u4))


class Generator(nn.Module):
    """完整生成器：CoarseNet → RefineNet → 融合输出。

    Args:
        cfg: model 子配置字典，含 coarse_in_channels / refine_in_channels / cbam_reduction
    """
    def __init__(self, cfg: dict = None):
        super().__init__()
        if cfg is None:
            cfg = {'coarse_in_channels': 3, 'refine_in_channels': 7, 'cbam_reduction': 16}
        self.coarse = CoarseNet(in_channels=cfg['coarse_in_channels'],
                                cbam_reduction=cfg['cbam_reduction'])
        self.refine = RefineNet(in_channels=cfg['refine_in_channels'])

    def forward(self, Iin: torch.Tensor, ms_gt: torch.Tensor = None):
        Ms, Mb, Ic4, Ic2, Ic1 = self.coarse(Iin)
        # ms_gt 不为 None 时启用 Teacher Forcing：用 GT 掩码引导 RefineNet（训练早期）
        ms_input = ms_gt if ms_gt is not None else Ms
        Ire = self.refine(torch.cat([Iin, ms_input, Ic1], dim=1))
        Icomp = Ire * Mb + Iin * (1 - Mb)
        return Ms, Mb, Ic4, Ic2, Ic1, Ire, Icomp
