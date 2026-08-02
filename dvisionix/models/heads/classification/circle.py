# -*- coding: utf-8 -*-
"""CircleLoss 度量学习头（归一化余弦，配合 CircleLoss 损失使用）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="circle_loss")
@HEADS.register(name="circle_loss_head")
class CircleLossHead(BaseModel):
    """Circle Loss 头部：归一化特征与类中心，输出 s*cos(theta)。

    与 NormFace 结构一致，但专为 ``loss: {type: circle_loss}`` 设计：
    损失在 logits（s*cos）上施加 circle 自适应 margin（目标类/非目标类分别调制）。
    """

    def __init__(self, in_channels, num_classes, s: float = 30.0):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        self.s = float(s)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_channels))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels=None):
        w_norm = F.normalize(self.weight, dim=1)
        x_norm = F.normalize(x, dim=1)
        return torch.mm(x_norm, w_norm.t()) * self.s


__all__ = ["CircleLossHead"]
