# -*- coding: utf-8 -*-
"""多标签分类头（MultiLabelHead）。"""

import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="multi_label")
class MultiLabelHead(BaseModel):
    """多标签分类头：逐标签 logits（配合 BCEWithLogits 损失）。"""

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


__all__ = ["MultiLabelHead"]
