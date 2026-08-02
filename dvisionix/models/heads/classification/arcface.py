# -*- coding: utf-8 -*-
"""ArcFace 度量学习头。"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="arcface")
class ArcFaceHead(BaseModel):
    """ArcFace 度量学习头：特征与类中心 L2 归一化，输出 s*cos(theta)。

    训练时若传入 ``labels``（forward(..., labels=...)）则施加 additive angular margin；
    完整 margin 训练可配合 ``loss: {type: arcface}``。
    """

    def __init__(self, in_channels, num_classes, s: float = 30.0, m: float = 0.5):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        cos_theta = torch.mm(x_norm, w_norm.t())

        if labels is not None and self.training:
            cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos_theta)
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            target_cos = torch.cos(theta + self.m)
            cos_theta = cos_theta * (1 - one_hot) + target_cos * one_hot

        return cos_theta * self.s


__all__ = ["ArcFaceHead"]
