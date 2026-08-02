# -*- coding: utf-8 -*-
"""NormFace 度量学习头（特征/权重 L2 归一化 + 缩放）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="normface")
@HEADS.register(name="normface_head")
class NormFaceHead(BaseModel):
    """NormFace：归一化特征与类中心后计算 s*cos(theta)（无 margin）。

    训练时可传 ``labels`` 保持与其它度量头一致的接口（当前无 margin 逻辑，labels 仅占位）。
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


__all__ = ["NormFaceHead"]
