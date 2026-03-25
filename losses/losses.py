"""
损失函数：LR Loss、SN Loss、Block Loss（Dice）、感知损失、风格损失、GAN Hinge Loss。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGG16Feature(nn.Module):
    """冻结的 VGG16 特征提取器，输出三个不同深度的特征层用于感知/风格损失。"""
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.feat1 = nn.Sequential(*vgg[:5])    # relu1_2
        self.feat2 = nn.Sequential(*vgg[5:10])  # relu2_2
        self.feat3 = nn.Sequential(*vgg[10:17]) # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        # 输入 [-1, 1] → [0, 1] → ImageNet 归一化
        x = (x + 1) / 2.0
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        f1 = self.feat1(x)
        f2 = self.feat2(f1)
        f3 = self.feat3(f2)
        return [f1, f2, f3]


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """计算 Gram 矩阵（归一化），用于风格损失。"""
    b, c, h, w = x.shape
    x = x.view(b, c, h * w)
    return torch.bmm(x, x.transpose(1, 2)) / (c * h * w)


class EnsExamLoss(nn.Module):
    """文字擦除任务全量损失，包含：LR Loss + 感知 + 风格 + SN + Block + GAN Hinge。

    Args:
        cfg: loss 子配置字典，所有权重均可从 config.yaml 调整，无需改代码
    """
    _DEFAULTS = {
        'lambda_lr': 1.0,
        'lambda_p': 0.05,
        'lambda_style': 120,
        'lambda_sn': 1.0,
        'lambda_b': 0.4,
        'lambda_n': [5, 6, 8, 10],
        'beta_n': [0.8, 0.8, 0.8, 2.0],
    }

    def __init__(self, cfg: dict = None):
        super().__init__()
        c = {**self._DEFAULTS, **(cfg or {})}
        self.vgg_feat    = VGG16Feature()
        self.lambda_n    = c['lambda_n']    # 多尺度文本区域权重
        self.beta_n      = c['beta_n']      # 多尺度非文本区域权重
        self.lambda_lr   = c['lambda_lr']
        self.lambda_p    = c['lambda_p']
        self.lambda_style = c['lambda_style']
        self.lambda_sn   = c['lambda_sn']
        self.lambda_b    = c['lambda_b']

    # ── 各分项损失 ────────────────────────────────────────────────────────

    def sn_loss(self, Ms: torch.Tensor, Ms_gt: torch.Tensor) -> torch.Tensor:
        """SN Loss：软笔画掩码 L1，按笔画稀疏度归一化，避免空白样本主导梯度。"""
        sum_gt = torch.sum(Ms_gt, dim=tuple(range(1, Ms.dim())))
        valid_mask = (sum_gt > 1.0).float()
        if valid_mask.sum() == 0:
            return Ms.sum() * 0
        l1_sum = torch.sum(torch.abs(Ms - Ms_gt), dim=tuple(range(1, Ms.dim())))
        sum_pred = torch.sum(Ms, dim=tuple(range(1, Ms.dim())))
        normalization = torch.min(sum_pred, sum_gt)
        loss_batch = l1_sum / (normalization + 1e-6)
        L_sn = (loss_batch * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        return L_sn

    def block_loss(self, Mb: torch.Tensor, Mb_gt: torch.Tensor) -> torch.Tensor:
        """Block Loss：Dice Loss，对文本块掩码形状监督，缓解正负样本不平衡。"""
        inter = (Mb * Mb_gt).sum([1, 2, 3])
        denom = (Mb ** 2).sum([1, 2, 3]) + (Mb_gt ** 2).sum([1, 2, 3])
        return (1 - 2 * inter / (denom + 1e-8)).mean()

    def lr_loss(self, Iouts: list, Igt_list: list, Mb_gt: torch.Tensor) -> torch.Tensor:
        """LR Loss：多尺度 L1，文本区域和非文本区域分别加权。"""
        loss = 0.0
        for i, (Iout, Igt) in enumerate(zip(Iouts, Igt_list)):
            _, _, h, w = Iout.shape
            mask = F.interpolate(Mb_gt, size=(h, w), mode='nearest')
            loss += (self.lambda_n[i] * F.l1_loss(Iout * mask, Igt * mask)
                     + self.beta_n[i] * F.l1_loss(Iout * (1 - mask), Igt * (1 - mask)))
        return loss

    def perceptual_loss(self, I_list: list, Igt: torch.Tensor) -> torch.Tensor:
        """感知损失：VGG 特征 L1，感知相似度比像素 L1 对纹理更敏感。"""
        gt_feats = self.vgg_feat(Igt)
        loss = 0.0
        for I in I_list:
            for f1, f2 in zip(self.vgg_feat(I), gt_feats):
                loss += F.l1_loss(f1, f2)
        return loss

    def style_loss(self, I_list: list, Igt: torch.Tensor) -> torch.Tensor:
        """风格损失：Gram 矩阵 L1，控制纹理和笔触风格一致性。"""
        gt_grams = [gram_matrix(f).detach() for f in self.vgg_feat(Igt)]
        loss = 0.0
        for I in I_list:
            for f_pred, G_gt in zip(self.vgg_feat(I), gt_grams):
                loss += torch.abs(gram_matrix(f_pred) - G_gt).sum(dim=(1, 2)).mean()
        return loss

    # ── GAN Hinge Loss ────────────────────────────────────────────────────

    @staticmethod
    def hinge_loss_D(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
        """判别器 Hinge Loss：鼓励真实样本 score > 1，生成样本 score < -1。"""
        return (F.relu(1.0 - real_score) + F.relu(1.0 + fake_score)).mean()

    @staticmethod
    def hinge_loss_G(fake_score: torch.Tensor) -> torch.Tensor:
        """生成器对抗损失：鼓励 score > -1，即让判别器对生成图打高分。"""
        return -F.relu(1.0 + fake_score).mean()

    # ── 总损失 ────────────────────────────────────────────────────────────

    def forward(self, gen_out: tuple, gt: tuple, disc_score: tuple):
        """
        Args:
            gen_out:    Generator 输出 (Ms, Mb, Ic4, Ic2, Ic1, Ire, Icomp)
            gt:         标签 (Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt)
            disc_score: 判别器分数 (global_score, local_score)

        Returns:
            L_total: 标量总损失
            parts:   各分项损失列表 [L_adv, L_lr, L_per, L_style, L_sn, L_block]
        """
        Ms, Mb, Ic4, Ic2, Ic1, Ire, Icomp = gen_out
        Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt = gt
        global_score, local_score = disc_score

        L_sn    = self.sn_loss(Ms, Ms_gt)            * self.lambda_sn
        L_block = self.block_loss(Mb, Mb_gt)         * self.lambda_b
        L_lr    = self.lr_loss([Ic4, Ic2, Ic1, Ire],
                               [Igt4, Igt2, Igt1, Igt], Mb_gt) * self.lambda_lr
        L_per   = self.perceptual_loss([Ire, Icomp], Igt) * self.lambda_p
        L_style = self.style_loss([Ire, Icomp], Igt)      * self.lambda_style
        L_adv   = (self.hinge_loss_G(global_score) + self.hinge_loss_G(local_score)) / 2

        L_total = L_adv + L_lr + L_per + L_style + L_sn + L_block
        return L_total, [L_adv, L_lr, L_per, L_style, L_sn, L_block]
