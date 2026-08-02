# -*- coding: utf-8 -*-
"""CosFace 度量学习头（additive cosine margin）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import BaseModel
from ....registry import HEADS


@HEADS.register()
@HEADS.register(name="cosface")
class CosFaceHead(BaseModel):
    """CosFace：logits = s * (cos(theta) - m)（目标类），其余 s * cos(theta)。"""

    def __init__(self, in_channels, num_classes, s: float = 30.0, m: float = 0.35):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels=None):
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        cos = torch.mm(x_norm, w_norm.t())
        if labels is not None and self.training:
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            cos = cos - one_hot * self.m
        return cos * self.s


__all__ = ["CosFaceHead"]