# -*- coding: utf-8 -*-
"""分类头：线性头（默认）、ArcFace（度量学习）、MultiLabel（多标签）。"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ...registry import HEADS


@HEADS.register()
@HEADS.register(name="cls_head")
@HEADS.register(name="linear_cls_head")
class ClsHead(BaseModel):
    """线性分类头：特征向量 -> logits (B, num_classes)。"""

    def __init__(self, in_channels, num_classes, dropout=0.0):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_channels, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


@HEADS.register()
@HEADS.register(name="arcface")
class ArcFaceHead(BaseModel):
    """ArcFace 度量学习头。

    输入特征与类中心向量均 L2 归一化，输出 ``s * cos(theta)``。
    训练时若传入 ``labels``（forward(..., labels=...)）则施加 additive angular margin。

    完整 margin 训练推荐配合 ``loss: {type: arcface}``（见 models/losses/classification.py）。
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
        cos_theta = torch.mm(x_norm, w_norm.t())  # (B, C) in [-1, 1]

        if labels is not None and self.training:
            cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos_theta)
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            target_cos = torch.cos(theta + self.m)
            cos_theta = cos_theta * (1 - one_hot) + target_cos * one_hot

        return cos_theta * self.s


@HEADS.register()
@HEADS.register(name="multi_label")
class MultiLabelHead(BaseModel):
    """多标签分类头：线性映射 + 逐标签 logits（配合 BCEWithLogits 损失）。"""

    def __init__(self, in_channels, num_classes, dropout=0.0):
        super().__init__(task_type="classification")
        self.num_classes = num_classes
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_channels, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


__all__ = ["ClsHead", "ArcFaceHead", "MultiLabelHead"]