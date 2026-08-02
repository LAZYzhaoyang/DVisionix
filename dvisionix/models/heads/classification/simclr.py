# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: SimCLR 对比学习投影头（MLP 投影，配合 InfoNCELoss）。
"""SimCLR 对比学习投影头（MLP 投影，配合 InfoNCELoss）。"""

import torch.nn as nn

from ....registry import HEADS
from ...base import BaseModel


@HEADS.register()
@HEADS.register(name="simclr")
@HEADS.register(name="simclr_head")
class SimCLRHead(BaseModel):
    """SimCLR 风格投影头：MLP（d -> hidden -> out_dim），把表示投影到对比空间。

    配合 ``loss: {type: info_nce}`` 做对比学习（双视角 InfoNCE）。
    """

    def __init__(self, in_channels, out_dim: int = 128, hidden: int = None, num_classes=None):
        super().__init__(task_type="classification")
        self.num_classes = num_classes  # 兼容装配器注入（投影头不使用）
        hidden = hidden or in_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, labels=None):
        """SimCLRHead 前向：特征 -> 投影向量 (B, proj_dim)。"""
        return self.proj(x)


__all__ = ["SimCLRHead"]
