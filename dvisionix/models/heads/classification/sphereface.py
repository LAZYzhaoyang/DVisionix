# -*- coding: utf-8 -*-
"""SphereFace 度量学习头（multiplicative angular margin）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="sphereface")
@HEADS.register(name="sphereface_head")
class SphereFaceHead(BaseModel):
    """SphereFace：目标类角度乘以 margin（cos(m*theta)）。"""

    def __init__(self, in_channels, num_classes, s: float = 30.0, m: float = 4):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels=None):
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        cos = torch.mm(x_norm, w_norm.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is not None and self.training:
            theta = torch.acos(cos)
            target_cos = torch.cos(self.m * theta)
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            cos = cos * (1 - one_hot) + target_cos * one_hot
        return cos * self.s


__all__ = ["SphereFaceHead"]
