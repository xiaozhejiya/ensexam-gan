"""
Reptile 元学习器，纯 PyTorch 实现，不依赖 learn2learn。

算法（每个 episode）：
    保存 meta 参数 θ_meta
    for i in range(n_tasks):
        从 θ_meta 出发，在 task_i 上跑 inner_steps 步 GAN → θ_i
        恢复 θ_meta
    θ_meta += ε · mean(θ_i − θ_meta)   # Reptile outer update

用 state_dict 深拷贝代替 clone_module，避免计算图残留导致的非叶节点问题。

DDP 兼容：
  - inner loop 不使用 DDP 包裹（需要反复 save/restore state_dict）
  - outer update 后通过 all-reduce 平均各 rank 的参数，等效于增大 n_tasks
"""
import copy

import torch
import torch.distributed as dist
from torch import optim

from losses.losses import EnsExamLoss
from train import unwrap_model, is_ddp


class ReptileMetaLearner:
    def __init__(self, G, D, criterion, device: torch.device, cfg: dict):
        self.G         = G
        self.D         = D
        self.criterion = criterion
        self.device    = device

        r = cfg['reptile']
        self.inner_lr    = r['inner_lr']
        self.inner_steps = r['inner_steps']
        self.meta_lr     = r['meta_lr']
        self.adam_betas  = tuple(cfg['train']['adam_betas'])

    def _inner_loop(self, task_loader):
        """
        在单个 task 上跑 inner_steps 步 GAN。

        流程：
          1. 深拷贝当前 meta 参数作为快照
          2. 直接在 meta 模型上做 inner 训练（参数为正常叶节点，优化器无问题）
          3. 深拷贝训练后的参数
          4. 用快照还原 meta 模型

        Returns:
            (G_task_state, D_task_state): 训练后的 state_dict 深拷贝
        """
        # 1. 保存 meta 参数快照
        G_meta_state = copy.deepcopy(unwrap_model(self.G).state_dict())
        D_meta_state = copy.deepcopy(unwrap_model(self.D).state_dict())

        # 2. 在 meta 模型上直接创建 inner 优化器（参数是正常叶节点）
        opt_G = optim.Adam(self.G.parameters(), lr=self.inner_lr, betas=self.adam_betas)
        opt_D = optim.Adam(self.D.parameters(), lr=self.inner_lr, betas=self.adam_betas)

        self.G.train()
        self.D.train()
        loader_iter = iter(task_loader)

        for _ in range(self.inner_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(task_loader)
                batch = next(loader_iter)

            Iin, Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt = [
                t.to(self.device) for t in batch
            ]
            gt = (Ms_gt, Mb_gt, Igt4, Igt2, Igt1, Igt)

            # 训练 D
            opt_D.zero_grad()
            gen_out        = self.G(Iin)
            Icomp          = gen_out[-1]
            real_g, real_l = self.D(Igt,            Mb_gt)
            fake_g, fake_l = self.D(Icomp.detach(), Mb_gt)
            loss_D = (EnsExamLoss.hinge_loss_D(real_g, fake_g)
                      + EnsExamLoss.hinge_loss_D(real_l, fake_l)) / 2
            loss_D.backward()
            opt_D.step()

            # 训练 G
            opt_G.zero_grad()
            fake_g, fake_l = self.D(Icomp, Mb_gt)
            loss_G, _      = self.criterion(gen_out, gt, (fake_g, fake_l))
            loss_G.backward()
            opt_G.step()

        # 3. 记录 inner 训练后的参数
        G_task_state = copy.deepcopy(unwrap_model(self.G).state_dict())
        D_task_state = copy.deepcopy(unwrap_model(self.D).state_dict())

        # 4. 还原 meta 参数，准备下一个 task
        unwrap_model(self.G).load_state_dict(G_meta_state)
        unwrap_model(self.D).load_state_dict(D_meta_state)

        return G_task_state, D_task_state, loss_G.item(), loss_D.item()

    def _reptile_update(self, model: torch.nn.Module, task_states: list):
        """θ_meta += ε · mean(θ_task_i − θ_meta)，只更新浮点参数。"""
        raw = unwrap_model(model)
        meta_state = raw.state_dict()
        for key in meta_state:
            if not meta_state[key].is_floating_point():
                continue  # 跳过 BatchNorm running_mean 等整数 buffer
            delta = torch.stack(
                [s[key].to(self.device) - meta_state[key] for s in task_states]
            ).mean(0)
            meta_state[key] = meta_state[key] + self.meta_lr * delta
        raw.load_state_dict(meta_state)

    def run_episode(self, task_loaders: list) -> dict:
        """对每个 task 做 inner loop，再做 Reptile outer update。

        Returns:
            stats: {'loss_G': float, 'loss_D': float}，所有 task 的平均损失
        """
        G_task_states, D_task_states = [], []
        sum_G = sum_D = 0.0
        for loader in task_loaders:
            g_state, d_state, lg, ld = self._inner_loop(loader)
            G_task_states.append(g_state)
            D_task_states.append(d_state)
            sum_G += lg
            sum_D += ld

        self._reptile_update(self.G, G_task_states)
        self._reptile_update(self.D, D_task_states)

        n = len(task_loaders)
        return {'loss_G': sum_G / n, 'loss_D': sum_D / n}

    def broadcast_params(self, src: int = 0):
        """DDP 模式下，all-reduce 所有 rank 的参数取平均，使各 rank 结果融合。

        每个 rank 独立采样不同 task 做 inner loop + outer update，
        all-reduce 后等效于 world_size × n_tasks_per_episode 的任务并行。
        比单纯 broadcast（丢弃其他 rank 工作）高效得多。
        """
        if not is_ddp():
            return
        world_size = dist.get_world_size()
        for model in (self.G, self.D):
            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data.div_(world_size)
            for buf in model.buffers():
                dist.all_reduce(buf.data, op=dist.ReduceOp.SUM)
                buf.data.div_(world_size)
