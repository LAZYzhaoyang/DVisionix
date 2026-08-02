# -*- coding: utf-8 -*-
"""分类任务损失：CrossEntropy（含 label smoothing）与 FocalLoss。"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry import LOSSES
from .base import BaseLoss


@LOSSES.register()
@LOSSES.register(name="cross_entropy")
class CrossEntropy(BaseLoss):
    """分类 / 分割通用的交叉熵损失（支持 ignore_index 与 label_smoothing）。"""

    name = "cross_entropy"

    def __init__(
        self,
        weight: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__(weight)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.criterion(logits, targets)


@LOSSES.register()
@LOSSES.register(name="focal")
class FocalLoss(BaseLoss):
    """Focal Loss：缓解类别不平衡，降低易分样本权重。

    支持 2D ``(B, C)`` 与 ND ``(B, C, ...)``（分割）输入。
    """

    name = "focal"

    def __init__(
        self,
        weight: float = 1.0,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(weight)
        self.alpha = alpha
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        num_classes = logits.shape[1]
        if logits.dim() > 2:
            logits = logits.permute(0, *range(2, logits.dim()), 1).reshape(-1, num_classes)
            targets = targets.reshape(-1)

        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        logits = logits[valid]
        targets = targets[valid]

        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        mod = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            mod = mod * alpha_t
        loss = mod * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


@LOSSES.register()
@LOSSES.register(name="binary_cross_entropy")
class BinaryCrossEntropy(BaseLoss):
    """多标签分类损失（BCE with logits，逐标签二分类）。"""

    name = "binary_cross_entropy"

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__(weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction=self.reduction)


@LOSSES.register()
@LOSSES.register(name="circle_loss")
class CircleLoss(BaseLoss):
    """Circle Loss：对目标类 / 非目标类相似度分别施加自适应 margin（Sun et al. 2020）。"""

    def __init__(
        self,
        gamma: float = 80.0,
        margin: float = 0.25,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.gamma = float(gamma)
        self.m = float(margin)
        self.o_p = 1.0 + margin
        self.o_n = -margin
        self.delta_p = 1.0 - margin
        self.delta_n = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        s = max(float(logits.abs().max().item()), 1.0)
        cos = logits / s  # 余弦相似度
        num_classes = logits.shape[1]
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        cos_p = (cos * one_hot).sum(dim=1)  # (N,)
        cos_n = cos * (1 - one_hot)  # (N, C)
        alpha_p = self.gamma * F.relu(self.o_p - cos_p)  # (N,)
        alpha_n = self.gamma * F.relu(cos_n - self.o_n)  # (N, C)
        neg_term = torch.logsumexp(alpha_n * (cos_n - self.delta_n), dim=1)  # (N,)
        pos_term = -alpha_p * (cos_p - self.delta_p)  # (N,)
        return (F.softplus(neg_term) + F.softplus(pos_term)).mean()


@LOSSES.register()
@LOSSES.register(name="info_nce")
class InfoNCELoss(BaseLoss):
    """InfoNCE 对比损失（SimCLR 风格）：双视角 z1/z2，对角线为正样本对。"""

    def __init__(self, temperature: float = 0.1, weight: float = 1.0):
        super().__init__(weight)
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = z1 @ z2.t() / self.temperature  # (B, B)
        labels = torch.arange(sim.shape[0], device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


__all__ = [
    "CrossEntropy",
    "FocalLoss",
    "BinaryCrossEntropy",
    "DistillationLoss",
    "CircleLoss",
    "InfoNCELoss",
]


@LOSSES.register()
@LOSSES.register(name="distillation")
class DistillationLoss(BaseLoss):
    """知识蒸馏损失：CE(硬标签) + alpha * KL(学生, 教师 soft targets)。

    用法：自定义 Task 的 training_step 中调用
    ``compute_loss(self.loss, student_logits, labels, teacher_logits=teacher_out)``。
    """

    name = "distillation"

    def __init__(self, weight: float = 1.0, alpha: float = 0.5, temperature: float = 4.0):
        super().__init__(weight)
        self.alpha = float(alpha)
        self.temperature = float(temperature)

    def forward(self, logits, targets, teacher_logits=None, **kwargs):
        t = self.temperature
        ce = F.cross_entropy(logits, targets)
        if teacher_logits is None:
            return ce
        kd = F.kl_div(
            F.log_softmax(logits / t, dim=-1),
            F.softmax(teacher_logits / t, dim=-1),
            reduction="batchmean",
        ) * (t * t)
        return (1.0 - self.alpha) * ce + self.alpha * kd
