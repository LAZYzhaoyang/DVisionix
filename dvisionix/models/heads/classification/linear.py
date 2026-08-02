# -*- coding: utf-8 -*-
"""线性分类头（ClsHead / LinearClsHead）。"""

import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel


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


__all__ = ["ClsHead"]
