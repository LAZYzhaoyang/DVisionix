# -*- coding: utf-8 -*-
# 作者: Zhaoyang Li
# 用途: 教学级网格检测头（DetHead）。
"""教学级网格检测头（DetHead）。

随 GridDetectionModel 一同用于演示/教学，生产请使用组件化检测头
（FCOSHead / RetinaNetHead / YOLOHead / DETRHead 等）。
"""

import torch.nn as nn

from ...registry import HEADS
from ..base import BaseModel


@HEADS.register()
@HEADS.register(name="det_head")
class DetHead(BaseModel):
    """单阶段网格检测头：每像素输出 objectness(1) + box(4) + 类别(num_classes)。

    Args:
        in_channels: 输入特征通道数。
        num_classes: 类别数。

    Returns:
        forward 输出 (B, 5 + num_classes, H, W) 的网格预测张量。
    """

    def __init__(self, in_channels, num_classes):
        super().__init__(task_type="detection")
        self.num_classes = num_classes
        self.out_channels = 5 + num_classes
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        """DetHead 前向：x -> 网格预测 (B, 5+num_classes, H, W)。"""
        if isinstance(x, (list, tuple)):
            x = x[-1]
        return self.conv(x)


__all__ = ["DetHead"]
