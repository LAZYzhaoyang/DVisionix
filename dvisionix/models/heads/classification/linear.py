# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 线性分类头（ClsHead / LinearClsHead）。
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

    def forward(self, x, labels=None):
        """ClsHead 前向：x -> logits (B, num_classes)。

        Args:
            x: 特征向量 (B, in_channels)。
            labels: 真实标签，**本头不使用**。接受它是为了与
                ``ClassificationTask`` / ``LinearClassifier`` 的 kwargs 透传契约
                保持一致 —— 训练步会无条件传 ``labels=``，普通分类头若拒绝该参数，
                最主流的 ``LinearClassifier + ClsHead + ClassificationTask`` 路径会在
                第一个 batch 直接 ``TypeError``。
        """
        return self.fc(x)


__all__ = ["ClsHead"]
