# -*- coding: utf-8 -*-
"""PartialFC 度量学习头（大规模类别采样子集 softmax，compact）。"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="partial_fc")
@HEADS.register(name="partial_fc_head")
class PartialFCHead(BaseModel):
    """PartialFC（compact）：每步仅对采样类别子集计算 logits，降低大规模类别显存/算力开销。

    - ``num_sample_classes=None``（默认）或 >= num_classes：等价于普通全量 softmax，返回完整 logits，
      可直接配合 LinearClassifier + ClassificationTask 使用。
    - ``num_sample_classes < num_classes`` 且训练时传 ``labels``：采样子集（必含 batch 内出现的类别），
      forward 返回 ``(logits_subset, sampled_indices)``，需配合自定义训练逻辑（见 ``remap_labels``）。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        s: float = 30.0,
        num_sample_classes: Optional[int] = None,
    ):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.s = float(s)
        self.num_sample_classes = num_sample_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        if (
            self.training
            and labels is not None
            and self.num_sample_classes
            and self.num_sample_classes < self.num_classes
        ):
            present = torch.unique(labels).tolist()
            if len(present) > self.num_sample_classes:
                raise ValueError(
                    f"num_sample_classes ({self.num_sample_classes}) 小于 batch 内类别数 ({len(present)})"
                )
            extra = self.num_sample_classes - len(present)
            pool = [c for c in range(self.num_classes) if c not in present]
            sampled = sorted(
                set(present) | set(torch.tensor(pool)[torch.randperm(len(pool))[:extra]].tolist())
            )
            w_sub = w_norm[sampled]  # (K, C)
            logits = torch.mm(x_norm, w_sub.t()) * self.s
            return logits, torch.as_tensor(sampled, dtype=torch.long, device=x.device)
        return torch.mm(x_norm, w_norm.t()) * self.s

    @staticmethod
    def remap_labels(labels: torch.Tensor, sampled_indices: torch.Tensor) -> torch.Tensor:
        """把全局类别 id 映射为采样子集内的局部索引。"""
        mapping = {int(idx): i for i, idx in enumerate(sampled_indices.tolist())}
        return torch.tensor(
            [mapping[int(lab)] for lab in labels.tolist()], dtype=torch.long, device=labels.device
        )


__all__ = ["PartialFCHead"]
